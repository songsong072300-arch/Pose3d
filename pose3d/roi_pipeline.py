"""Two-instance RGB video pose pipeline.

Detector -> per-instance crop -> shared corner network -> PnP -> EMA smoothing.
The detector is supplied by Ultralytics YOLO; the remaining stages are pure
OpenCV/PyTorch and can be tested independently.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np, torch
from PIL import Image
from .boxdreamer_single import BoxDreamerSingle

class RoiPoseDataset(torch.utils.data.Dataset):
    """Derive single-instance ROI samples from existing BOP annotations."""
    def __init__(self, root, split='train', pad=.15):
        import json
        from pathlib import Path
        self.root=Path(root); meta=json.loads((self.root/'corner_labels/annotations.json').read_text()); sp=json.loads((self.root/'corner_labels/splits.json').read_text()); ids=set(sp[split]); self.items=[]
        for r in meta['records']:
            if r['id'] not in ids: continue
            for ins in r['instances']: self.items.append((r,ins))
        self.pad=pad
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        r,ins=self.items[i]; im=Image.open(self.root/r['image']).convert('RGB'); W,H=im.size; p=np.asarray(ins['corners_2d'],np.float32); x1,y1=p.min(0); x2,y2=p.max(0); dx,dy=(x2-x1)*self.pad,(y2-y1)*self.pad; xa,ya,xb,yb=max(0,x1-dx),max(0,y1-dy),min(W,x2+dx),min(H,y2+dy); crop=im.crop((xa,ya,xb,yb)).resize((240,180),Image.Resampling.BILINEAR); x=torch.from_numpy(np.asarray(crop,np.float32)).permute(2,0,1)/255.; x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        q=(p-[xa,ya])*[240/(xb-xa),180/(yb-ya)]; yy,xx=np.mgrid[0:180,0:240]; h=np.zeros((8,180,240),np.float32); sigma=max(2.,min((x2-x1)*240/(xb-xa),(y2-y1)*180/(yb-ya))/18.)
        for c,(u,v) in enumerate(q): h[c]=np.exp(-((xx-u)**2+(yy-v)**2)/(2*sigma*sigma))
        return {'image':x,'heatmap':torch.from_numpy(h),'corners':torch.from_numpy(q),'id':r['id']+'_'+str(ins['instance_id'])}

@dataclass
class PoseResult:
    box: np.ndarray
    corners: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    score: float

def extract_peaks(hm, threshold=0.15):
    """Return one sub-pixel peak per heatmap channel (H,W heatmap)."""
    pts=[]
    for c in hm:
        _, mx, _, loc = cv2.minMaxLoc(c.astype(np.float32))
        x,y=loc
        if mx < threshold: pts.append([np.nan,np.nan]); continue
        # 3x3 weighted centroid improves quantization over argmax.
        x0,y0=max(0,x-1),max(0,y-1); p=c[y0:min(c.shape[0],y+2),x0:min(c.shape[1],x+2)]
        yy,xx=np.mgrid[y0:y0+p.shape[0],x0:x0+p.shape[1]]; w=np.maximum(p,0)
        pts.append([float((xx*w).sum()/max(w.sum(),1e-6)),float((yy*w).sum()/max(w.sum(),1e-6))])
    return np.asarray(pts,np.float32)

class EMATracker:
    def __init__(self, alpha=.65, max_jump=180): self.alpha,self.max_jump,self.prev=alpha,max_jump,[]
    def update(self, results):
        results=sorted(results,key=lambda r: float(r.box[0]))
        out=[]
        for i,r in enumerate(results):
            if i < len(self.prev) and np.linalg.norm(r.box[:2]-self.prev[i].box[:2]) < self.max_jump:
                if r.rvec is not None and self.prev[i].rvec is not None:
                    r.rvec=self.alpha*r.rvec+(1-self.alpha)*self.prev[i].rvec
                    r.tvec=self.alpha*r.tvec+(1-self.alpha)*self.prev[i].tvec
            out.append(r)
        self.prev=out; return out

class VideoPosePipeline:
    def __init__(self, corner_weights, object_points, camera_matrix, dist=None,
                 detector_weights=None, device="auto", crop_pad=.15):
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends,"mps",None) is not None and torch.backends.mps.is_available() else "cpu") if device=="auto" else device
        self.net=BoxDreamerSingle().to(self.device).eval()
        ck=torch.load(corner_weights,map_location=self.device); self.net.load_state_dict(ck.get("model",ck),strict=True)
        self.obj=np.asarray(object_points,np.float32); self.K=np.asarray(camera_matrix,np.float32); self.dist=np.zeros(5,np.float32) if dist is None else np.asarray(dist,np.float32)
        self.crop_pad=crop_pad; self.tracker=EMATracker()
        if detector_weights:
            from ultralytics import YOLO
            self.detector=YOLO(detector_weights)
        else: self.detector=None

    def _detect(self, frame):
        if self.detector is None: raise RuntimeError("detector_weights is required for video inference")
        r=self.detector.predict(frame,verbose=False,device=self.device)[0]
        boxes=r.boxes.xyxy.detach().cpu().numpy(); scores=r.boxes.conf.detach().cpu().numpy()
        ids=np.argsort(scores)[::-1][:2]
        return [(boxes[i],float(scores[i])) for i in ids]

    def __call__(self, frame):
        H,W=frame.shape[:2]; results=[]
        for box,score in self._detect(frame):
            x1,y1,x2,y2=box; pw,ph=(x2-x1)*self.crop_pad,(y2-y1)*self.crop_pad
            xa,ya,xb,yb=max(0,int(x1-pw)),max(0,int(y1-ph)),min(W,int(x2+pw)),min(H,int(y2+ph))
            crop=cv2.cvtColor(frame[ya:yb,xa:xb],cv2.COLOR_BGR2RGB)
            im=Image.fromarray(crop).resize((240,180),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.asarray(im,np.float32)).permute(2,0,1)/255.
            x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            with torch.no_grad(): hm=self.net(x[None].to(self.device))[0].cpu().numpy()
            p=extract_peaks(hm); p[:,0]*=(xb-xa)/240.; p[:,1]*=(yb-ya)/180.; p += [xa,ya]
            ok=np.isfinite(p).all(1)
            rv=tv=None
            if ok.sum()>=4:
                ok2,rv,tv=cv2.solvePnPRansac(self.obj[ok],p[ok],self.K,self.dist,flags=cv2.SOLVEPNP_SQPNP)
                if not ok2: rv=tv=None
            results.append(PoseResult(np.asarray(box),p,rv,tv,score))
        return self.tracker.update(results)

def draw_results(frame, results):
    for r in results:
        x1,y1,x2,y2=r.box.astype(int); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        for x,y in r.corners:
            if np.isfinite(x): cv2.circle(frame,(int(x),int(y)),3,(0,0,255),-1)
        if r.tvec is not None: cv2.putText(frame,f"z={r.tvec[2,0]:.0f}mm",(x1,y1-6),0,.6,(0,255,0),2)
    return frame
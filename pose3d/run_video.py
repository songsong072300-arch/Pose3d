import argparse, cv2, json, numpy as np
from .roi_pipeline import VideoPosePipeline, draw_results

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',required=True); ap.add_argument('--detector',required=True); ap.add_argument('--corner-weights',required=True); ap.add_argument('--camera',required=True); ap.add_argument('--out',default='outputs/pose.mp4'); ap.add_argument('--model-points',default='TripoSR/dataset/demo-bin-picking/models/models_info.json'); args=ap.parse_args()
    cam=json.load(open(args.camera)); K=np.asarray(cam.get('cam_K',cam),np.float32).reshape(3,3)
    info=json.load(open(args.model_points)); i=info.get('1',info.get(1)); sx,sy,sz=[i[k] for k in ('size_x','size_y','size_z')]
    # 合成渲染若放大过模型（render_config.json），PnP 的 3D 角点须同步缩放。
    import pathlib
    rc=pathlib.Path(args.model_points).parent.parent/'render_config.json'
    model_scale=float(json.load(open(rc)).get('model_scale',1.0)) if rc.exists() else 1.0
    sx,sy,sz=sx*model_scale,sy*model_scale,sz*model_scale
    obj=np.array([[-sx/2,-sy/2,-sz/2],[-sx/2,-sy/2,sz/2],[-sx/2,sy/2,-sz/2],[-sx/2,sy/2,sz/2],[sx/2,-sy/2,-sz/2],[sx/2,-sy/2,sz/2],[sx/2,sy/2,-sz/2],[sx/2,sy/2,sz/2]],np.float32)
    pipe=VideoPosePipeline(args.corner_weights,obj,K,detector_weights=args.detector)
    cap=cv2.VideoCapture(args.video); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(3)); h=int(cap.get(4)); out=cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));
    while True:
        ok,f=cap.read()
        if not ok: break
        out.write(draw_results(f,pipe(f)))
    cap.release(); out.release(); print(args.out)
if __name__=='__main__': main()

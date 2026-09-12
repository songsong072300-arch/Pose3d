"""Run only the two-object detection stage on one image."""
import argparse, cv2, numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); ap.add_argument('--weights',required=True); ap.add_argument('--out',default='outputs/detections.jpg'); ap.add_argument('--conf',type=float,default=.25); args=ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('未安装 ultralytics，请在 pose3d 环境执行: pip install ultralytics') from e
    im=cv2.imread(args.image)
    if im is None: raise SystemExit(f'无法读取图像: {args.image}')
    model=YOLO(args.weights); r=model.predict(im,conf=args.conf,verbose=False,device='mps' if hasattr(__import__('torch').backends,'mps') and __import__('torch').backends.mps.is_available() else 'cpu')[0]
    boxes=r.boxes.xyxy.detach().cpu().numpy(); scores=r.boxes.conf.detach().cpu().numpy(); order=np.argsort(scores)[::-1][:2]
    for n,i in enumerate(order):
        x1,y1,x2,y2=boxes[i].astype(int); cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3); cv2.putText(im,f'ACTION4 {scores[i]:.2f}',(x1,max(20,y1-8)),0,.7,(0,255,0),2); print(n,boxes[i].tolist(),float(scores[i]))
    import pathlib; pathlib.Path(args.out).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(args.out,im); print('saved',args.out)
if __name__=='__main__': main()

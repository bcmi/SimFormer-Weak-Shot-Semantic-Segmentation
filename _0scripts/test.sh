# -------------------------------------------------------- COCO Stuff 10K ------------------------------------------------------------------------------------
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s1_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/model_final_cb03eb_COCO.pkl OUTPUT_PREFIX Fully_COCO_S1
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s2_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/model_final_cb03eb_COCO.pkl OUTPUT_PREFIX Fully_COCO_S2
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s3_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/model_final_cb03eb_COCO.pkl OUTPUT_PREFIX Fully_COCO_S3
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s4_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/model_final_cb03eb_COCO.pkl OUTPUT_PREFIX Fully_COCO_S4

python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s1_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/Release/COCO/SimFormer_S1.pth OUTPUT_PREFIX os_COCO_S1

python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s1_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/Release/COCO/final_S1.pth OUTPUT_PREFIX Ours_COCO_S1
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s2_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/Release/COCO/final_S2.pth OUTPUT_PREFIX Ours_COCO_S2
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s3_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/Release/COCO/final_S3.pth OUTPUT_PREFIX Ours_COCO_S3
python train_net_prop.py --config-file _1Prop_Cfgs/coco_sutff_10k/s4_seg.yaml --eval-only MODEL.WEIGHTS ../../pretrained/Release/COCO/final_S4.pth OUTPUT_PREFIX Ours_COCO_S4
# -------------------------------------------------------- ADE 20K ------------------------------------------------------------------------------------
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s1_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/model_final_d8dbeb_ADE.pkl OUTPUT_PREFIX Fully_ADE_S1
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s2_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/model_final_d8dbeb_ADE.pkl OUTPUT_PREFIX Fully_ADE_S2
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s3_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/model_final_d8dbeb_ADE.pkl OUTPUT_PREFIX Fully_ADE_S3
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s4_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/model_final_d8dbeb_ADE.pkl OUTPUT_PREFIX Fully_ADE_S4

python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s1_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/Release/ADE/final_S1.pth OUTPUT_PREFIX Ours_ADE_S1
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s2_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/Release/ADE/final_S2.pth OUTPUT_PREFIX Ours_ADE_S2
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s3_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/Release/ADE/final_S3.pth OUTPUT_PREFIX Ours_ADE_S3
python train_net_prop.py --config-file _1Prop_Cfgs/ade20k-150/s4_seg.yaml --eval-only MODEL.WEIGHTS  ../../pretrained/Release/ADE/final_S4.pth OUTPUT_PREFIX Ours_ADE_S4

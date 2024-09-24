
_base_ = '/home/muyishen2040/periodontal_mmdet/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'


dataset_type = 'CocoDataset'
data_root = '/home/muyishen2040/periodontal_mmdet/dataset/taipei_medical/tmu_coco/'  

metainfo = {
    'classes': ('tooth', 'bone_level_area', 'cej_level'),  
    'palette': [(220, 20, 60), (0, 255, 0), (0, 0, 255)]  
}


train_dataloader = dict(
    batch_size=4,  
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_annotations.json',  
        data_prefix=dict(img=data_root),  
        metainfo=metainfo
    )
)


val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'val_annotations.json',  
        data_prefix=dict(img=data_root),  
        metainfo=metainfo
    )
)


test_dataloader = val_dataloader


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3),  
        mask_head=dict(num_classes=3)    
    )
)



optimizer = dict(type='SGD', lr=0.02 / 8, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',  
    warmup='linear',  
    warmup_iters=500,  
    warmup_ratio=0.001,  
    step=[8, 11]  
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)


evaluation = dict(
    interval=1,  
    metric=['segm']  
)


log_config = dict(
    interval=50,  
    hooks=[
        dict(type='TextLoggerHook')  
    ]
)


checkpoint_config = dict(
    interval=1  
)


work_dir = './work_dirs/tmu_mask_rcnn/'  

val_evaluator = dict(ann_file=data_root + 'val_annotations.json')
test_evaluator = val_evaluator

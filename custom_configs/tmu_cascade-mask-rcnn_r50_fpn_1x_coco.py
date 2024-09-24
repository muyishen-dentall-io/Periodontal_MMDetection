_base_ = '/home/muyishen2040/periodontal_mmdet/mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

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
        metainfo=metainfo,
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
        bbox_head=[dict(type='Shared2FCBBoxHead', num_classes=3) for _ in range(3)],  
        mask_head=dict(type='FCNMaskHead', num_classes=3)  
    )
)




lr_config = dict(
    policy='step',  
    warmup='linear',  
    warmup_iters=500,  
    warmup_ratio=0.001,  
    step=[5, 10, 15]  
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)


evaluation = dict(
    interval=2,  
    metric=['segm']  
)


log_config = dict(
    interval=50,  
    hooks=[
        dict(type='TextLoggerHook')  
    ]
)


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=4)
)


work_dir = './work_dirs/tmu_cascade_mask_rcnn/'  


val_evaluator = dict(ann_file=data_root + 'val_annotations.json')
test_evaluator = val_evaluator

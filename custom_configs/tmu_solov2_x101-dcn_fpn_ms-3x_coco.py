_base_ = '/home/muyishen2040/periodontal_mmdet/mmdetection/configs/solov2/solov2_x101-dcn_fpn_ms-3x_coco.py'

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
    mask_head=dict(
        num_classes=3  
    )
)

# optimizer = dict(type='SGD', lr=0.01 / 8, momentum=0.9, weight_decay=0.0001)  

lr_config = dict(
    policy='step',  
    warmup='linear',  
    warmup_iters=500,  
    warmup_ratio=0.001,  
    step=[7, 10]  
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)


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


work_dir = './work_dirs/tmu_solov2/'  


val_evaluator = dict(ann_file=data_root + 'val_annotations.json')
test_evaluator = val_evaluator


_base_ = '/home/muyishen2040/periodontal_mmdet/mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'


num_things_classes = 3  
num_stuff_classes = 0  
num_classes = num_things_classes + num_stuff_classes

dataset_type = 'CocoDataset'

data_root = '/home/muyishen2040/periodontal_mmdet/dataset/taipei_medical/tmu_coco/'


model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes
    )
)

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


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_annotations.json',
    metric=['bbox', 'segm']  
)

test_evaluator = val_evaluator


work_dir = './work_dirs/tmu_mask2former/'  

import torch
class loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def IOU(self, box1, target_box):
        x_axis_indices = torch.arange(box1.shape[1]).view(1,-1,1,1).to(box1.device)
        y_axis_indices = torch.arange(box1.shape[2]).view(1,1,-1,1).to(box1.device)
        x_axis_indices = x_axis_indices.expand_as(box1)
        y_axis_indices = y_axis_indices.expand(box1.shape[0],box1.shape[1],box1.shape[2],1)

        box1[...,0] = box1[...,0]*64+y_axis_indices[...,0]*64
        box1[...,1] = box1[...,1]*64+x_axis_indices[...,0]*64
        box1[...,2] = box1[...,2]*448
        box1[...,3] = box1[...,3]*448

        box1[...,0] -= box1[...,2]/2
        box1[...,1] -= box1[...,3]/2
        box1[...,2] += box1[...,0]
        box1[...,3] += box1[...,1]

        target_box[...,1] = target_box[...,1]*64+y_axis_indices[...,0]*64
        target_box[...,2] = target_box[...,2]*64+x_axis_indices[...,0]*64
        target_box[...,3] = target_box[...,3]*448
        target_box[...,4] = target_box[...,4]*448

        target_box[...,1] -= target_box[...,3]/2
        target_box[...,2] -= target_box[...,4]/2
        target_box[...,3] += target_box[...,1]
        target_box[...,4] += target_box[...,2]

        x1_min = torch.max(box1[..., 0], target_box[..., 1])
        y1_min = torch.max(box1[..., 1], target_box[..., 2])
        x2_max = torch.min(box1[..., 2], target_box[..., 3])
        y2_max = torch.min(box1[..., 3], target_box[..., 4])

        inter_area = torch.clamp(x2_max - x1_min, min=0) * torch.clamp(y2_max - y1_min, min=0)
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        target_box_area = (target_box[..., 3] - target_box[..., 1]) * (target_box[..., 4] - target_box[..., 2])

        iou = inter_area / (box1_area + target_box_area - inter_area)
        return iou

    def error(self, box, target):
        loss = 0
        x_loss = 5*torch.sum((box[...,0] - target[...,1])**2)
        y_loss = 5*torch.sum((box[...,1] - target[...,2])**2)
        width_loss = 5*torch.sum((box[...,2] - target[...,3])**2)
        height_loss = 5*torch.sum((box[...,3] - target[...,4])**2)
        confidence_loss = torch.sum((box[...,4] - target[...,0])**2)
        loss = x_loss + y_loss + width_loss + height_loss + confidence_loss

        return loss

    def noobj_error(self, box):
        loss = torch.sum(box[...,4]**2)
        return loss

    def forward(self, outputs, targets):
        loss = 0

        obj_mask = (targets[...,0] == 1)
        noobj_mask = (targets[...,0] == 0)
        obj_mask = obj_mask.unsqueeze(-1).expand_as(outputs)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(outputs)

        obj_outputs = torch.where(obj_mask, outputs, torch.tensor(0))
        noobj_outputs = torch.where(noobj_mask, outputs, torch.tensor(0))

        loss += torch.sum((obj_outputs[...,10:30]-targets[...,10:30])**2)

        boxes_set1 = obj_outputs[...,[0,1,2,3,4]]
        boxes_set2 = obj_outputs[...,[5,6,7,8,9]]
        boxes_targets = targets[...,[0,1,2,3,4]]

        boxes_iou1 = self.IOU(boxes_set1.clone(), boxes_targets.clone())
        boxes_iou2 = self.IOU(boxes_set2.clone(), boxes_targets.clone())
        iou_mask = boxes_iou1 > boxes_iou2
        loss+=self.error(torch.where(iou_mask.unsqueeze(-1), boxes_set1, boxes_set2),boxes_targets)
        loss+=self.noobj_error(torch.where(~iou_mask.unsqueeze(-1), boxes_set1, boxes_set2))

        return loss

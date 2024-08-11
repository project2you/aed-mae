import argparse
import datetime
import json
import os
import time
from pathlib import Path

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from engine_train import train_one_epoch, test_one_epoch
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc
import torch

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # กำหนดให้ใช้ GPU ถ้าพร้อมใช้งาน
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # โหลด dataset train ถ้า run_type เป็น train
    if args.run_type == 'train':
        dataset_train = AbnormalDatasetGradientsTrain(args)
        print(dataset_train)
        print(f"Number of training samples: {len(dataset_train)}")

        # ตรวจสอบว่ามี training samples หรือไม่
        if len(dataset_train) == 0:
            raise ValueError("No training samples found, please check the dataset path and files.")

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=0,  # ตั้งค่า num_workers เป็น 0 เพื่อป้องกันปัญหา worker process
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # โหลด dataset test สำหรับทั้ง train และ inference
    dataset_test = AbnormalDatasetGradientsTest(args)
    print(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=0,  # ตั้งค่า num_workers เป็น 0
        pin_memory=args.pin_mem, drop_last=False,
    )

    # กำหนด model ตาม dataset
    if args.dataset == 'avenue':
        model = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                abnormal_score_func=args.abnormal_score_func,
                                masking_method=args.masking_method,
                                grad_weighted_loss=args.grad_weighted_rec_loss).float()
    else:
        model = mae_cvt_patch8(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                               use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                               abnormal_score_func=args.abnormal_score_func,
                               masking_method=args.masking_method,
                               grad_weighted_loss=args.grad_weighted_rec_loss).float()
    
    # ย้ายโมเดลไปยัง GPU
    model.to(device)

    # เลือกทำ train หรือ inference
    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model)
    elif args.run_type == "inference":
        student = torch.load(args.output_dir + "/checkpoint-best-student.pth", map_location=device)['model']
        teacher = torch.load(args.output_dir + "/checkpoint-best.pth", map_location=device)['model']
        for key in student:
            if 'student' in key:
                teacher[key] = student[key]
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)


def do_training(args, data_loader_test, data_loader_train, device, log_writer, model):
    print("actual lr: %.2e" % args.lr)

    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_micro = 0.0
    best_micro_student = 0.0
    test_stats = None  # กำหนดค่าเริ่มต้นให้กับ test_stats

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args
        )
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if len(data_loader_test) > 0:
            test_stats = test_one_epoch(
                model, data_loader_test, device, epoch, log_writer=log_writer, args=args
            )
            log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}
        else:
            print("Warning: No data in data_loader_test.")
            log_stats_test = {}

        if args.output_dir:
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, latest=True)
        if test_stats and test_stats.get('micro', 0) > best_micro:
            best_micro = test_stats['micro']
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, best=True)
        if args.start_TS_epoch <= epoch:
            if test_stats and test_stats.get('micro', 0) > best_micro_student:
                best_micro_student = test_stats['micro']
                misc.save_model(args=args, model=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best=True, student=True)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            if log_stats_test:
                with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats_test) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    args = parser.parse_args()

    if args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

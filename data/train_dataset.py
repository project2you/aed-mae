import glob
import os
import random
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetGradientsTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        
        self.percent_abnormal = args.percent_abnormal
        self.input_3d = args.input_3d
        self.data, self.gradients, self.masks_abnormal = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        gradients = []
        masks_abnormal = []
        extension = None

        # ตรวจสอบว่ามีไฟล์ที่รองรับในพาธหรือไม่
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "training/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break

        if extension is None:
            raise ValueError(f"No supported image files found in {os.path.join(data_path, 'training/frames')}")

        # ดึงข้อมูลจากโฟลเดอร์ training/frames
        dirs = list(glob.glob(os.path.join(data_path, "training", "frames", "*")))
        if not dirs:
            raise ValueError(f"No directories found in {os.path.join(data_path, 'training/frames')}")

        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            if not imgs_path:
                print(f"No images found in directory {dir}")
                continue

            data += imgs_path
            video_name = os.path.basename(dir)
            gradients_path = []
            for img_path in imgs_path:
                frame_no = int(os.path.basename(img_path).split('.')[0])
                gradients_path.append(os.path.join(data_path, "training", "gradients2", video_name,
                                                   f"{frame_no:04d}.jpg"))
                masks_abnormal.append(os.path.join(data_path, "training", "masks_abnormal", video_name,
                                                   f"{frame_no:04d}.jpg"))
            gradients += gradients_path

        if len(data) == 0:
            raise ValueError(f"No training samples found in {data_path}")

        print(f"Number of training samples: {len(data)}")
        return data, gradients, masks_abnormal

    def __getitem__(self, index):
        random_uniform = random.uniform(0, 1)
        img = cv2.imread(self.data[index])
        dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
        previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
        next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
        
        if self.input_3d:
            img = np.concatenate([previous_img, img, next_img], axis=-1)
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # สร้าง mask แบบ 2 มิติ
        gradient = cv2.imread(self.gradients[index])
        target = cv2.imread(self.data[index])

        # ตรวจสอบมิติของ img, gradient, และ mask
        if img.shape[:2] != self.args.input_size or gradient.shape[:2] != self.args.input_size:
            img = cv2.resize(img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])
            mask = cv2.resize(mask, self.args.input_size[::-1])

        if target.shape[:2] != self.args.input_size:
            target = cv2.resize(target, self.args.input_size[::-1])

        # เพิ่มมิติใหม่ให้กับ mask เพื่อทำให้มี 3 มิติ (ตรงกับ target)
        mask = np.expand_dims(mask, axis=-1)
        
        # รวม target และ mask
        target = np.concatenate((target, mask), axis=-1)

        # ทำการปรับมิติของ img, gradient, และ target ให้เข้ากับ PyTorch
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        
        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        
        target = (target - 127.5) / 127.5
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        
        return img, gradient, target

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=4):
        frame_path = os.path.join(dir_path, f"{frame_no + direction:04d}.jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(dir_path, f"{frame_no:04d}.jpg")
        return cv2.imread(frame_path) if os.path.exists(frame_path) else None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__

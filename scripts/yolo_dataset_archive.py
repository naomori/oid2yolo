import os
import sys
import glob
import tarfile
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

root_dir = sys.argv[1]

train_list = ["train_" + str(i) for i in range(10)]
train_list.extend(["train_" + chr(i) for i in range(97, 97+6)])
tarball_names = train_list
tarball_names.append("val")
tarball_names.append("test")
tarballs = [root_dir + "/" + tarball_name + ".tar.gz" for tarball_name in tarball_names]

train_labels = [root_dir + f"/labels/train/{str(i)}*.txt" for i in range(10)]     # 0-9
train_labels.extend([root_dir + f"/labels/train/{chr(i)}*.txt" for i in range(97, 97+6)])    # a-f
tarball_labels = train_labels
tarball_labels.append(root_dir + "/labels/val")
tarball_labels.append(root_dir + "/labels/test")

train_images = [root_dir + f"/images/train/{str(i)}*.jpg" for i in range(10)]     # 0-9
train_images.extend([root_dir + f"/images/train/{chr(i)}*.jpg" for i in range(97, 97+6)])    # a-f
tarball_images = train_images
tarball_images.append(root_dir + "/images/val")
tarball_images.append(root_dir + "/images/test")

arc_dir = "yolo"


def archive(tarball, labels, images):
    with tarfile.open(tarball, 'w:gz') as tar:
        [tar.add(label, arcname=label.replace(root_dir, arc_dir)) for label in glob.glob(labels)]
        [tar.add(image, arcname=image.replace(root_dir, arc_dir)) for image in glob.glob(images)]
    stat_info = os.stat(tarball)
    return tarball, stat_info.st_size


futures = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for tarball_name, label_list, image_list in zip(tarballs, tarball_labels, tarball_images):
        future = executor.submit(archive, tarball_name, label_list, image_list)
        futures.append(future)
concurrent.futures.wait(futures, timeout=None)
tarinfo_list = [future.result() for future in concurrent.futures.as_completed(futures)]
for tar_name, tar_size in tarinfo_list:
    print(f"{tar_name}: {tar_size}[bytes]")

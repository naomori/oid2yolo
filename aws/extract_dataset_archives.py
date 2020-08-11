import os
import subprocess
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


def create_tarball_list():
    tarball_list = [f"train_{i}.tar.gz" for i in range(10)]
    tarball_list += [f"train_{chr(ord('a') + i)}.tar.gz" for i in range(6)]
    tarball_list += ["val.tar.gz"]
    tarball_list += ["test.tar.gz"]
    return tarball_list


def extract_tarball(src_dir, tar_name, dst_dir):
    subprocess.run(f"rsync -ahv ${src_dir}/${tar_name} ${dst_dir}/", shell=True)
    subprocess.run(f"chmod 660 ${dst_dir}/${tar_name}", shell=True)
    subprocess.run(f"tar -C ${dst_dir} -zxf ${dst_dir}/${tar_name}", shell=True)
    subprocess.run(f"rm -f ${dst_dir}/${tar_name}", shell=True)


source_dir = "/s3/yolo-nc2"
destination_dir = "/content/yolov5/data"

futures = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for tarball_name in create_tarball_list():
        future = executor.submit(extract_tarball, source_dir, tarball_name, destination_dir)
        futures.append(future)
concurrent.futures.wait(futures, timeout=None)
[future.result() for future in concurrent.futures.as_completed(futures)]

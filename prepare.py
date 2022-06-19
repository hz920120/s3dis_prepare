import glob
import numpy as np
import h5py


def convert_txt_to_h5(source = r"D:\AnacondaCode\04Deep_Learning\03 3D point cloud\Pointnet_Pointnet2_pytorch-master\data\s3dis\alter_s3dis_my",
                      target = r"D:\AnacondaCode\04Deep_Learning\03 3D point cloud\data\S3DIS_hdf5"):

    for file in glob.glob(source+"/*.txt"):
        name = file.replace('\\', '/').split("/")[-1][:-4]
        data = np.loadtxt(file)
        points = data[:, :6]
        labels = data[:, 6]

        f = h5py.File(target+"/"+name+".h5", "w")
        f.create_dataset("data", data=points)
        f.create_dataset("label", data=labels)
        f.close()

convert_txt_to_h5()
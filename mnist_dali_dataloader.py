import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

@pipeline_def(batch_size=64, num_threads=8, exec_async=False, exec_pipelined=False)
def mnist_pipeline(data_dir=None, file_list=None, mode="training", crop_w=28, crop_h=28,
                   shard_id=0, num_shards=1):
    
    file_names, labels = fn.readers.file(
        file_root=data_dir,
        file_list=file_list,
        random_shuffle=(mode == 'training'),
        name="Reader",
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True)

    images = fn.decoders.image(
        file_names,
        device="mixed",
        output_type=types.GRAY)

    if mode == "training":
        images = fn.rotate(images, angle=fn.random.uniform(range=(-7.0, 7.0)), keep_size=True)

        images = fn.noise.gaussian(images, stddev=0.01)

        images = fn.brightness_contrast(images, brightness=1.1, contrast=1.2)

        # Random crop and translate
        # Since MNIST images are 28x28, we use a smaller crop size to allow for effective random cropping
        # and then resize back to 28x28 to maintain the original dimensionality.
        images = fn.random_resized_crop(
            images,
            size=(crop_w, crop_h),
            device="gpu",
            random_area=[0.8, 1.0],  # Controls the area of the input to be cropped
            random_aspect_ratio=[0.9, 1.1])  # Controls the aspect ratio of the crop

    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(crop_w, crop_h),
        mean=[128.0],
        std=[128.0],
        mirror=fn.random.coin_flip(probability=0.0))

    return labels, images

class CustomDALIIterator(dali_torch.DALIGenericIterator):
    def __init__(self, pipelines, *args, **kwargs):
        super(CustomDALIIterator, self).__init__(pipelines, ["labels", "images"], *args, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]

        labels = out["labels"]
        images = out["images"]

        return labels, images

class MNISTDataLoader:
    def __init__(self, batch_size, device_id, num_threads, seed, data_dir=None, file_list=None, mode='training',
                 crop_w=28, crop_h=28, shard_id=0, num_shards=1):
        self.pipeline = mnist_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            data_dir=data_dir,
            file_list=file_list,
            mode=mode,
            crop_w=crop_w,
            crop_h=crop_h)
        self.pipeline.build()
        self.loader = CustomDALIIterator(
            [self.pipeline],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)

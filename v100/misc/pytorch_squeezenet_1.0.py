import torch
import torchvision.models
import time

from torchvision.models.squeezenet import model_urls

batch_size = 128
n_channels = 3
n_rows = 224
n_cols = 224
n_batches = 128
data_shape = (batch_size, n_channels, n_rows, n_cols)


def measure_squeezenet_throughput():
    model_urls['squeezenet1_0'] = \
    model_urls['squeezenet1_0'].replace('https://', 'http://')

    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.cuda()
    model.eval()

    synth_data = [
        torch.rand(data_shape).cuda(non_blocking=True)
        for _ in range(n_batches)
    ]
    with torch.no_grad():
        start_time = time.time()
        for sd_ in synth_data:
            _ = model(sd_)

        # time elapsed: 4.469737768173218
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("time elapsed: {}".format(elapsed_time))
        print("time per batch: {}, batch size: {}".format(
            elapsed_time / n_batches, batch_size
        ))


if __name__ == "__main__":
    measure_squeezenet_throughput()

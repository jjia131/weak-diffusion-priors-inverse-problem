import matplotlib.pyplot as plt


def plot_images(batch, noisy_batch=None, path='test.jpg', show=False, scale_factor=None,titles=None, col_num =8, title_fontsize=12):
    batch_size = batch.shape[0]
    channel = batch.shape[1]
    img_height, img_width = batch.shape[2], batch.shape[3]

    if noisy_batch is not None:
        noisy_batch_size = noisy_batch.shape[0]
        total_images = batch_size + noisy_batch_size
    else:
        total_images = batch_size


    cols = max(min(total_images, col_num),2)
    rows = (total_images + cols - 1) // cols  # Calculate required rows

    if scale_factor is None:
        if img_height <= 50:
            scale_factor = 2
        else:
            scale_factor = 3

    fig, axs = plt.subplots(rows, cols, figsize=(cols * scale_factor, rows * scale_factor))
    axs = axs.flatten()

    if channel == 3:
        cmap = "viridis"
    else:
        cmap = "gray"

    for i in range(batch_size):
        # Display batch images
        axs[i].imshow(batch[i].permute(1, 2, 0).cpu().numpy(), cmap=cmap)
        axs[i].axis('off')
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i], fontsize=title_fontsize)

    if noisy_batch is not None:
        for i in range(noisy_batch_size):
            # Display noisy batch images
            axs[batch_size + i].imshow(noisy_batch[i].permute(1, 2, 0).cpu().numpy(), cmap=cmap)
            axs[batch_size + i].axis('off')
            if titles is not None and (batch_size + i) < len(titles):
                axs[batch_size + i].set_title(titles[batch_size + i])

    # Turn off unused axes
    for i in range(total_images, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)


def plot_images_numpy(batch, noisy_batch=None, path='test.jpg', show=False, scale_factor=None,titles=None, col_num =8):
    batch_size = batch.shape[0]
    channel = batch.shape[1]
    img_height, img_width = batch.shape[2], batch.shape[3]

    if noisy_batch is not None:
        total_images = batch_size * 2
    else:
        total_images = batch_size

    cols = max(min(total_images, col_num),2)
    rows = (total_images + cols - 1) // cols  # Calculate required rows

    if scale_factor is None:
        if img_height <= 50:
            scale_factor = 2
        else:
            scale_factor = 3    

    fig, axs = plt.subplots(rows, cols, figsize=(cols * scale_factor, rows * scale_factor))
    axs = axs.flatten()

    if channel == 3:
        cmap = "viridis"
    else:
        cmap = "gray"

    for i in range(batch_size):
        # Display batch images
        axs[i].imshow(batch[i].transpose(1, 2, 0), cmap=cmap)
        axs[i].axis('off')
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i])

    if noisy_batch is not None:
        for i in range(batch_size):
            # Display noisy batch images
            axs[batch_size + i].imshow(noisy_batch[i].transpose(1, 2, 0), cmap=cmap)
            axs[batch_size + i].axis('off')
            if titles is not None and (batch_size + i) < len(titles):
                axs[batch_size + i].set_title(titles[batch_size + i])

    # Turn off unused axes
    for i in range(total_images, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)
import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels = false;
env.useBrowserCache = true;

let model = null;
let processor = null;

async function load() {
  self.postMessage({ type: "progress", message: "downloading..." });

  [model, processor] = await Promise.all([
    AutoModel.from_pretrained("briaai/RMBG-1.4", {
      config: { model_type: "custom" },
      dtype: "fp32",
    }),
    AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
      config: {
        do_normalize: true,
        do_pad: false,
        do_rescale: true,
        do_resize: true,
        image_mean: [0.5, 0.5, 0.5],
        image_std: [1, 1, 1],
        resample: 2,
        rescale_factor: 0.00392156862745098,
        size: { width: 1024, height: 1024 },
      },
    }),
  ]);
}

async function removeBg(id, dataUrl) {
  const image = await RawImage.fromURL(dataUrl);

  const { pixel_values } = await processor(image);

  // output shape is [1,1,H,W] — squeeze to [H,W] then unsqueeze to [1,H,W] for RawImage
  const result = await model({ input: pixel_values });
  const outputTensor = result.output ?? Object.values(result)[0];
  const maskData = (
    await RawImage.fromTensor(
      outputTensor.squeeze().unsqueeze(0).mul(255).to("uint8"),
    ).resize(image.width, image.height)
  ).data;

  // Build RGBA directly from RawImage pixel data (avoids second fetch which can fail)
  const { width, height, channels } = image;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; ++i) {
    rgba[i * 4] = image.data[i * channels];
    rgba[i * 4 + 1] = image.data[i * channels + 1];
    rgba[i * 4 + 2] = image.data[i * channels + 2];
    rgba[i * 4 + 3] = maskData[i];
  }

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);

  const outBlob = await canvas.convertToBlob({ type: "image/png" });
  const buffer = await outBlob.arrayBuffer();
  self.postMessage({ type: "done", id, buffer }, [buffer]);
}

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    try {
      await load();
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", id: -1, error: err.message });
    }
  } else if (data.type === "process") {
    try {
      await removeBg(data.id, data.dataUrl);
    } catch (err) {
      self.postMessage({ type: "error", id: data.id, error: err.message });
    }
  }
};

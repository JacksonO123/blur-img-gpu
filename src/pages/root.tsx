import { ChangeEvent, createState, onPageMount } from "@jacksonotto/lampjs";
import "./root.css";

const computeCode = `
struct ImageSize {
  size: vec2<u32>
}

struct Image {
  rgba: array<i32>
}

struct Radius {
  amount: i32
}

@group(0) @binding(0)
var<storage, read> imageSize: ImageSize;

@group(0) @binding(1)
var<storage, read> imageData: Image;

@group(0) @binding(2)
var<storage, read> blurRadius: Radius;

@group(0) @binding(3)
var<storage, read_write> imageOut: Image;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index: u32 = global_id.x * 4 + global_id.y * imageSize.size.x * 4;

  var r: i32 = 0;
  var g: i32 = 0;
  var b: i32 = 0;
  var a: i32 = 0;

  var counter: i32 = 0;

  for (var x: i32 = -blurRadius.amount; x <= blurRadius.amount; x++) {
    for (var y: i32 = -blurRadius.amount; y <= blurRadius.amount; y++) {
      let pixelIndex = (i32(index) - (x * 4) - (y * i32(imageSize.size.x) * 4));

      if (
        i32(global_id.x) + x >= 0 &&
        i32(global_id.x) + x <= i32(imageSize.size.x) &&
        i32(global_id.y) + y >= 0 &&
        i32(global_id.y) + y <= i32(imageSize.size.y)
      ) {
        r += imageData.rgba[pixelIndex];
        g += imageData.rgba[pixelIndex + 1];
        b += imageData.rgba[pixelIndex + 2];
        a += imageData.rgba[pixelIndex + 3];
        counter++;
      }
    }
  }

  imageOut.rgba[index] = r / counter;
  imageOut.rgba[index + 1] = g / counter;
  imageOut.rgba[index + 2] = b / counter;
  imageOut.rgba[index + 3] = a / counter;
}
`;

const Root = () => {
  const blurAmount = createState(10);
  let img: HTMLImageElement | null = null;
  let canvas: HTMLCanvasElement | null = null;
  let ctx: CanvasRenderingContext2D | null = null;
  let device: GPUDevice | null = null;
  let bindGroupLayout: GPUBindGroupLayout | null = null;
  let shaderModule: GPUShaderModule | null = null;

  onPageMount(async () => {
    canvas = document.querySelector("#canvas") as HTMLCanvasElement | null;
    const image = document.querySelector("#img") as HTMLImageElement | null;
    if (!image || !canvas) return;

    ctx = canvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (!ctx) return;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    device = await adapter.requestDevice();

    bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        } as GPUBindGroupLayoutEntry,
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        } as GPUBindGroupLayoutEntry,
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        } as GPUBindGroupLayoutEntry,
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        } as GPUBindGroupLayoutEntry,
      ],
    });

    shaderModule = device.createShaderModule({
      code: computeCode,
    });

    image.onload = async () => {
      img = image;
      console.log("loaded");
      blurImage(image, blurAmount().value);
    };
  });

  async function blurImage(image: HTMLImageElement, blurRadius: number) {
    // console.log(canvas, ctx, device, bindGroupLayout, shaderModule);
    if (!canvas || !ctx || !device || !bindGroupLayout || !shaderModule) return;

    const width = image.width;
    const height = image.height;
    canvas.width = width;
    canvas.height = height;

    ctx.drawImage(image, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const tempPixelData = imageData.data;
    const pixelData = new Int32Array(tempPixelData.length);
    for (let i = 0; i < pixelData.length; i++) {
      pixelData[i] = tempPixelData[i];
    }

    const sizeArray = new Int32Array([width, height]);
    const gpuSizeArray = device.createBuffer({
      mappedAtCreation: true,
      size: sizeArray.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    const arraySizeBuffer = gpuSizeArray.getMappedRange();
    new Int32Array(arraySizeBuffer).set(sizeArray);
    gpuSizeArray.unmap();

    const gpuPixelData = device.createBuffer({
      mappedAtCreation: true,
      size: pixelData.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    const arrayPixelBuffer = gpuPixelData.getMappedRange();
    new Int32Array(arrayPixelBuffer).set(pixelData);
    gpuPixelData.unmap();

    const gpuResultBuffer = device.createBuffer({
      size: pixelData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const gpuReadBuffer = device.createBuffer({
      size: pixelData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const blurRadiusArray = new Int32Array([blurRadius]);
    const gpuBlurRadius = device.createBuffer({
      mappedAtCreation: true,
      size: blurRadiusArray.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    const blurRadiusBuffer = gpuBlurRadius.getMappedRange();
    new Int32Array(blurRadiusBuffer).set(blurRadiusArray);
    gpuBlurRadius.unmap();

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: gpuSizeArray,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: gpuPixelData,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: gpuBlurRadius,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: gpuResultBuffer,
          },
        },
      ],
    });

    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(width, height);
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
      gpuResultBuffer,
      0,
      gpuReadBuffer,
      0,
      pixelData.byteLength
    );
    device.queue.submit([commandEncoder.finish()]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const res = new Int32Array(
      gpuReadBuffer.getMappedRange(0, pixelData.byteLength)
    );

    const newData = new Uint8ClampedArray(pixelData.length);
    for (let i = 0; i < res.length; i++) {
      newData[i] = res[i];
    }

    const newImageData = new ImageData(newData, width, height);
    ctx.putImageData(newImageData, 0, 0);
  }

  const handleBlurChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = +e.currentTarget.value;
    blurAmount(Math.max(0, value));
  };

  const handleBlurImage = () => {
    if (img !== null) {
      blurImage(img, blurAmount().value);
    }
  };

  return (
    <div>
      <div class="controls">
        <h1>Blur image</h1>
        <div class="group">
          <input
            type="number"
            value={blurAmount()}
            onChange={handleBlurChange}
          />
          <button onClick={handleBlurImage}>Blur</button>
        </div>
      </div>
      <img id="img" src="/image.jpeg" hidden />
      <canvas id="canvas" />
    </div>
  );
};

export default Root;

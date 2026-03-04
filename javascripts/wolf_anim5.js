import {
  mat4,
  vec3,
  quat,
} from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js'

export class WolfAnim extends HTMLElement {
  constructor() {
    super()
    this.attachShadow({ mode: 'open' })
    this.shadowRoot.innerHTML = `      <style>
      :host {
        display: flex;
        justify-content: center;
        align-items: center;  
        width: 96vw;
        height: 96vh;
        position: relative;
        overflow: hidden;
      }
      </style>
      <canvas width="400" height="400" id="gpu-canvas" "></canvas>`
  }

  connectedCallback() {
    this.main()
  }

  async main() {
    // --- 1. SHADER WGSL ---
    const shaderCode = `
      struct Uniforms {
          modelViewMatrix: mat4x4f,
          projectionMatrix: mat4x4f,
      }
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> boneMatrices: array<mat4x4f>;
      
      struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) color: vec3f,
      }
      
      @vertex
      fn vs_main(
          @location(0) position: vec3f,
          @location(1) joints: vec4u,
          @location(2) weights: vec4f
      ) -> VertexOutput {
          let skinMatrix = 
              weights.x * boneMatrices[joints.x] +
              weights.y * boneMatrices[joints.y] +
              weights.z * boneMatrices[joints.z] +
              weights.w * boneMatrices[joints.w];
      
          var out: VertexOutput;
          let worldPos = uniforms.modelViewMatrix * skinMatrix * vec4f(position, 1.0);
          out.pos = uniforms.projectionMatrix * worldPos;
          //out.color = vec3f(1.0, 0.5, 0.0) * (position.y + 0.6); 
          out.color = vec3f(1.0, 1.0, 1.0) * (position.y + 0.6);
          return out;
      }

      @fragment
      fn fs_main(@location(0) color: vec3f) -> @location(0) vec4f {
          return vec4f(color, 1.0);
      }
    `

    // --- 2. SETUP WEBGPU ---
    const canvas = this.shadowRoot.getElementById('gpu-canvas')
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter.requestDevice()
    const context = canvas.getContext('webgpu')
    const format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format, alphaMode: 'premultiplied' })

    // --- 3. ÎNCĂRCARE DATE GLTF ---
    const response = await fetch('./images/wolf.gltf')
    const gltf = await response.json()
    const binResponse = await fetch(`./images/${gltf.buffers[0].uri}`)
    const binaryData = await binResponse.arrayBuffer()

    function getBufferData(accessorIndex) {
      if (accessorIndex === undefined) return null
      const accessor = gltf.accessors[accessorIndex]
      const bufferView = gltf.bufferViews[accessor.bufferView]
      const typeSize = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[
        accessor.type
      ]
      let TypedArray = Float32Array
      if (accessor.componentType === 5123) TypedArray = Uint16Array
      if (accessor.componentType === 5121) TypedArray = Uint8Array
      return new TypedArray(
        binaryData,
        (bufferView.byteOffset || 0) + (accessor.byteOffset || 0),
        accessor.count * typeSize
      )
    }

    const primitive = gltf.meshes[0].primitives[0]
    const skin = gltf.skins[0]
    const animation = gltf.animations[0]

    const positions = getBufferData(primitive.attributes.POSITION)
    const joints = getBufferData(primitive.attributes.JOINTS_0)
    const weights = getBufferData(primitive.attributes.WEIGHTS_0)
    const ibmDataRaw = getBufferData(skin.inverseBindMatrices)

    // --- 4. BUFFERE GPU ---
    const vertexBuffer = device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    })
    new Float32Array(vertexBuffer.getMappedRange()).set(positions)
    vertexBuffer.unmap()

    const jointsBuffer = device.createBuffer({
      size: joints.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    })
    new Uint32Array(jointsBuffer.getMappedRange()).set(new Uint32Array(joints))
    jointsBuffer.unmap()

    const weightsBuffer = device.createBuffer({
      size: weights.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    })
    new Float32Array(weightsBuffer.getMappedRange()).set(weights)
    weightsBuffer.unmap()

    const boneCount = skin.joints.length
    const boneMatricesData = new Float32Array(boneCount * 16)
    const boneStorageBuffer = device.createBuffer({
      size: boneMatricesData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const uniformBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const msaaTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      sampleCount: 4, // Trebuie să fie același cu cel din pipeline
      format: format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    // De asemenea, asigură-te că depthTexture are și ea sampleCount: 4
    const depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      sampleCount: 4, // Modifică aici din 1 în 4
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    const shaderModule = device.createShaderModule({ code: shaderCode })
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 16,
            attributes: [{ shaderLocation: 1, offset: 0, format: 'uint32x4' }],
          },
          {
            arrayStride: 16,
            attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
      multisample: {
        count: 4,
      },
    })

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: boneStorageBuffer } },
      ],
    })

    // --- 5. LOGICA DE ANIMAȚIE (wgpu-matrix) ---
    const inverseBindMatrices = []
    for (let i = 0; i < boneCount; i++) {
      inverseBindMatrices.push(ibmDataRaw.subarray(i * 16, i * 16 + 16))
    }

    function interpolate(channel, time) {
      const sampler = animation.samplers[channel.sampler]
      const times = getBufferData(sampler.input)
      const values = getBufferData(sampler.output)

      let i = 0
      while (i < times.length - 2 && time >= times[i + 1]) i++
      const t = (time - times[i]) / (times[i + 1] - times[i])

      if (channel.target.path === 'rotation') {
        return quat.slerp(
          values.subarray(i * 4, i * 4 + 4),
          values.subarray((i + 1) * 4, (i + 1) * 4 + 4),
          t
        )
      } else {
        return vec3.lerp(
          values.subarray(i * 3, i * 3 + 3),
          values.subarray((i + 1) * 3, (i + 1) * 3 + 3),
          t
        )
      }
    }

    const globalNodeMatrices = new Array(gltf.nodes.length)
    let maxTime = 0
    animation.channels.forEach((c) => {
      const times = getBufferData(animation.samplers[c.sampler].input)
      maxTime = Math.max(maxTime, times[times.length - 1])
    })

    function computeTransforms(nodeIndex, parentMat, time) {
      const node = gltf.nodes[nodeIndex]

      let t = node.translation
        ? vec3.create(...node.translation)
        : vec3.create(0, 0, 0)
      let r = node.rotation
        ? quat.create(...node.rotation)
        : quat.create(0, 0, 0, 1)
      let s = node.scale ? vec3.create(...node.scale) : vec3.create(1, 1, 1)

      animation.channels.forEach((channel) => {
        if (channel.target.node === nodeIndex) {
          const val = interpolate(channel, time)
          if (channel.target.path === 'translation') t = val
          if (channel.target.path === 'rotation') r = val
          if (channel.target.path === 'scale') s = val
        }
      })

      // --- CONSTRUCȚIE MANUALĂ MATRICE (FĂRĂ funcția problematică) ---
      const localMat = mat4.identity()
      // Ordinea corectă: Translație * Rotație * Scalare
      mat4.translate(localMat, t, localMat)

      // Convertim cuaternionul de rotație într-o matrice 4x4 și o înmulțim
      const rotationMat = mat4.fromQuat(r)
      mat4.multiply(localMat, rotationMat, localMat)

      mat4.scale(localMat, s, localMat)
      // --------------------------------------------------------------

      const globalMat = mat4.multiply(parentMat, localMat)
      globalNodeMatrices[nodeIndex] = globalMat

      if (node.children) {
        node.children.forEach((child) =>
          computeTransforms(child, globalMat, time)
        )
      }
    }

    // --- 6. RENDER LOOP ---
    const frame = (timestamp) => {
      const time = (timestamp / 1000) % maxTime
      const rootNode = skin.skeleton ?? gltf.scenes[0].nodes[0]

      computeTransforms(rootNode, mat4.identity(), time)

      for (let i = 0; i < boneCount; i++) {
        const globalMat = globalNodeMatrices[skin.joints[i]]
        const finalMat = mat4.multiply(globalMat, inverseBindMatrices[i])
        boneMatricesData.set(finalMat, i * 16)
      }
      device.queue.writeBuffer(boneStorageBuffer, 0, boneMatricesData)

      // const modelView = mat4.identity()
      // mat4.translate(modelView, [0, -1.0, -4.0], modelView)
      // mat4.rotateY(modelView, timestamp * 0.0005, modelView)
      // mat4.scale(modelView, [0.02, 0.02, 0.02], modelView)

      const modelView = mat4.identity()
      mat4.translate(modelView, [0, -1.0, -4.0], modelView)
      mat4.rotateY(modelView, Math.PI / 2, modelView)
      // Am șters linia cu rotateY
      mat4.scale(modelView, [0.016, 0.016, 0.016], modelView)

      const projection = mat4.perspective(
        (45 * Math.PI) / 180,
        canvas.width / canvas.height,
        0.1,
        100.0
      )

      device.queue.writeBuffer(uniformBuffer, 0, modelView)
      device.queue.writeBuffer(uniformBuffer, 64, projection)

      const commandEncoder = device.createCommandEncoder()
      const textureView = context.getCurrentTexture().createView()
      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: msaaTexture.createView(), // Randăm în textura multisample
            resolveTarget: context.getCurrentTexture().createView(), // Rezultatul merge în canvas
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 0.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })

      renderPass.setPipeline(pipeline)
      renderPass.setBindGroup(0, bindGroup)
      renderPass.setVertexBuffer(0, vertexBuffer)
      renderPass.setVertexBuffer(1, jointsBuffer)
      renderPass.setVertexBuffer(2, weightsBuffer)
      renderPass.draw(positions.length / 3)
      renderPass.end()

      device.queue.submit([commandEncoder.finish()])
      requestAnimationFrame(frame)
    }
    requestAnimationFrame(frame)
  }
}
customElements.define('wolf-anim5', WolfAnim)

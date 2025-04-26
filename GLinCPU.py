import numpy as np
from PIL import Image
from datetime import datetime

class MockGPU:
  def __init__(self):
    self.vertex_buffer = None
    self.vertex_normals = None
    self.modelview_matrix = None
    self.projection_matrix = None
    self.normal_matrix = None
    self.viewport = None

    self.gl_Position = None  # equivalent to gl_Position
    self.ndc_vertices = None
    self.vertPos = None         # varying vec3 vertPos (world or view space)
    self.normalInterp = None    # varying vec3 normalInterp (world or view space)

    self.framebuffer = None

  def upload_vertex_buffer(self, vertices):
    self.vertex_buffer = vertices

  def upload_vertex_normals(self, normals):
    self.vertex_normals = normals

  def upload_modelview_matrix(self, modelview):
    self.modelview_matrix = modelview

  def upload_projection_matrix(self, projection):
    self.projection_matrix = projection

  def upload_normal_matrix(self, normalmat):
    self.normal_matrix = normalmat

  def set_viewport(self, x, y, width, height):
    self.viewport = (x, y, width, height)

  def vertex_shader_stage(self):
    if (self.vertex_buffer is None or self.vertex_normals is None or
        self.modelview_matrix is None or self.projection_matrix is None or
        self.normal_matrix is None):
      raise ValueError("Vertex buffer, normals, and matrices must be set before shading.")

    # -------- Simulate GLSL vertex shader --------

    # inputPosition
    positions = self.vertex_buffer[:, :3]

    # inputNormal
    normals = self.vertex_normals

    # modelview * vec4(inputPosition, 1.0)
    modelview_positions = (self.modelview_matrix @ np.hstack((positions, np.ones((positions.shape[0], 1)))).T).T

    # projection * modelview * vec4(inputPosition, 1.0)
    gl_Position = (self.projection_matrix @ modelview_positions.T).T

    # perspective divide
    ndc_vertices = gl_Position[:, :3] / gl_Position[:, 3][:, np.newaxis]

    # vertPos = vec3(modelview * inputPosition) / w
    vertPos = modelview_positions[:, :3] / modelview_positions[:, 3][:, np.newaxis]

    # normalInterp = normalMat * vec4(inputNormal, 0.0)
    normalInterp = (self.normal_matrix @ np.hstack((normals, np.zeros((normals.shape[0], 1)))).T).T[:, :3]

    # normalize normals (good practice)
    normalInterp = normalInterp / np.linalg.norm(normalInterp, axis=1, keepdims=True)

    # -------- Store results in \"GPU memory\" --------
    self.gl_Position = gl_Position
    self.ndc_vertices = ndc_vertices
    self.vertPos = vertPos
    self.normalInterp = normalInterp

    print("Clip Space Positions (gl_Position):")
    print(gl_Position)

    print("\nNDC Vertices (after perspective divide):")
    print(ndc_vertices)

    print("\nInterpolated vertex positions (vertPos):")
    print(vertPos)

    print("\nInterpolated normals (normalInterp):")
    print(normalInterp)
  
  def rasterize_and_fragment_shading(self):
    if (self.viewport is None or
        self.gl_Position is None or
        self.vertPos is None or
        self.normalInterp is None):
      raise ValueError("Viewport, gl_Position, vertPos, or normalInterp not set.")

    x0, y0, w, h = self.viewport

    # 1. Map NDC to screen space
    def ndc_to_screen(ndc):
      ndc_x, ndc_y = ndc[0], ndc[1]
      screen_x = int(x0 + (ndc_x + 1) * 0.5 * w)
      screen_y = int(y0 + (ndc_y + 1) * 0.5 * h)
      return np.array([screen_x, screen_y])

    # Use gl_Position after perspective divide to get NDC
    ndc_vertices = self.gl_Position[:, :3] / self.gl_Position[:, 3][:, np.newaxis]
    screen_vertices = np.array([ndc_to_screen(v) for v in ndc_vertices])

    # 2. Compute bounding box
    min_x = max(np.min(screen_vertices[:, 0]), 0)
    max_x = min(np.max(screen_vertices[:, 0]), w - 1)
    min_y = max(np.min(screen_vertices[:, 1]), 0)
    max_y = min(np.max(screen_vertices[:, 1]), h - 1)

    # 3. Prepare intermediate fragment buffer
    self.final_color = np.zeros((h, w, 3), dtype=np.float32)  # for output
    self.fragment_mask = np.zeros((h, w), dtype=bool)
    
    # 4. Constants for fragment shader
    lightPos = np.array([0.0, 0.0, 3.0])
    lightPos = self.modelview_matrix @ np.hstack((lightPos, 1))
    lightPos = lightPos[:3] / lightPos[3] 
    lightColor = np.array([1.0, 1.0, 1.0])
    lightPower = 4.0
    ambientColor = np.array([0.1, 0.1, 0.1])
    diffuseColor = np.array([1.0, 0.0, 0.0])
    specColor = np.array([1.0, 1.0, 1.0])
    shininess = 16.0
    screenGamma = 2.2

    # 5. Rasterization
    def edge(p0, p1, p):
      return (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0])

    v0, v1, v2 = screen_vertices
    attr_pos0, attr_pos1, attr_pos2 = self.vertPos
    attr_nrm0, attr_nrm1, attr_nrm2 = self.normalInterp

    area = edge(v0, v1, v2)
    if area == 0:
      print("Degenerate triangle (area = 0)")
      return

    for y in range(min_y, max_y + 1):
      for x in range(min_x, max_x + 1):
        p = np.array([x, y]) + np.array([0.5, 0.5])  # Center pixel

        w0 = edge(v1, v2, p)
        w1 = edge(v2, v0, p)
        w2 = edge(v0, v1, p)

        if w0 >= 0 and w1 >= 0 and w2 >= 0:
          # Inside triangle
          w0 /= area
          w1 /= area
          w2 /= area

          # Interpolate attributes
          interp_vertPos = w0 * attr_pos0 + w1 * attr_pos1 + w2 * attr_pos2
          interp_normalInterp = w0 * attr_nrm0 + w1 * attr_nrm1 + w2 * attr_nrm2

          # --- Fragment Shader Stage (Blinn-Phong Lighting) ---
          normal = interp_normalInterp / np.linalg.norm(interp_normalInterp + 1e-8)

          lightDir = lightPos - interp_vertPos
          distance2 = np.dot(lightDir, lightDir)
          lightDir = lightDir / np.linalg.norm(lightDir + 1e-8)

          lambertian = max(np.dot(lightDir, normal), 0.0)
          specular = 0.0

          if lambertian > 0.0:
            viewDir = -interp_vertPos
            viewDir = viewDir / np.linalg.norm(viewDir + 1e-8)

            halfDir = (lightDir + viewDir)
            halfDir = halfDir / np.linalg.norm(halfDir + 1e-8)

            specAngle = max(np.dot(halfDir, normal), 0.0)
            specular = specAngle ** shininess

          colorLinear = ambientColor + \
                        diffuseColor * lambertian * lightColor * lightPower / distance2 + \
                        specColor * specular * lightColor * lightPower / distance2

          # Gamma correction
          colorGammaCorrected = np.power(np.clip(colorLinear, 0.0, 1.0), 1.0 / screenGamma)

          # Store final color
          self.final_color[h - 1 - y, x, :] = colorGammaCorrected
          self.fragment_mask[h - 1 - y, x] = True


  def blend_color(self):
    if self.final_color is None or self.fragment_mask is None:
      raise ValueError("No final color or mask available for blending.")

    h, w, _ = self.final_color.shape

    # Start from black background
    blended_framebuffer = np.zeros((h, w, 3), dtype=np.float32)

    # Copy the computed color where the triangle covers
    for y in range(h):
      for x in range(w):
        if self.fragment_mask[y, x]:
          blended_framebuffer[y, x, :] = self.final_color[y, x, :]

    self.framebuffer = blended_framebuffer
  
  def save_image(self, base_filename="output", add_timestamp=True):
    # Save image with timestamp
    if self.framebuffer is None:
      raise ValueError("No framebuffer to save.")

    # Normalize framebuffer values to [0,255] for PNG
    img_data = np.clip(self.framebuffer * 255.0, 0, 255).astype(np.uint8)

    # Create PIL Image
    img = Image.fromarray(img_data, mode="RGB")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{base_filename}_{timestamp}.png" if add_timestamp else f"{base_filename}.png"

    # Save image
    img.save(filename)
    print(f"Saved image to {filename}")


if __name__ == "__main__":
  # Example setup
  SCREEN_WIDTH = 250
  SCREEN_HEIGHT = 250
  vertices = np.array([
      [-2.0, -1.0, 0.0, 1.0],  # Vertex A
      [ 2.0, -1.0, 0.0, 1.0],  # Vertex B
      [ 0.0,  2.0, 0.0, 1.0],  # Vertex C
  ])
  vertex_normals = np.array([
    [0.0, 0.0, 1.0],  # Vertex A
    [0.0, 0.0, 1.0],  # Vertex B
    [0.0, 0.0, 1.0],  # Vertex C
  ])
  
  # MVP matrix (assuming you have M, V, P from earlier)
  M = np.eye(4)
  V = np.array([
      [1.0, 0.0, 0.0,  0.0],
      [0.0, 1.0, 0.0,  0.0],
      [0.0, 0.0, 1.0, -5.0],
      [0.0, 0.0, 0.0,  1.0],
  ])
  fov_deg = 60
  aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
  near = 0.1
  far = 100.0
  f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
  P = np.array([
      [f / aspect_ratio, 0.0, 0.0,                               0.0],
      [0.0,              f,   0.0,                               0.0],
      [0.0,              0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
      [0.0,              0.0, -1.0,                              0.0],
  ])
  
  gpu = MockGPU()
  gpu.upload_vertex_buffer(vertices)
  gpu.upload_vertex_normals(vertex_normals)
  gpu.upload_projection_matrix(P)
  gpu.set_viewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
  delta_rad_Y = np.pi / 2.2 / 50
  for timestep in range(50):
    modelview = V @ M
  
    # Extract upper-left 3x3 matrix (ignore translation)
    modelview3x3 = modelview[:3, :3]

    # Then compute normalMat as inverse-transpose of the 3x3
    normalMat3x3 = np.linalg.inv(modelview3x3).T

    # Expand to 4x4 if needed (for compatibility with 4D vectors)
    normalMat = np.eye(4)
    normalMat[:3, :3] = normalMat3x3

    # Use MockGPU
    gpu.upload_modelview_matrix(modelview)
    gpu.upload_normal_matrix(normalMat)
    gpu.vertex_shader_stage()
    gpu.rasterize_and_fragment_shading()
    gpu.blend_color()
    gpu.save_image("triangle_render" + str(timestep).zfill(4), add_timestamp=False)
    
    # Rotate around Y-axis
    rotation_matrix = np.array([
      [np.cos(delta_rad_Y), 0.0, np.sin(delta_rad_Y), 0.0],
      [0.0,                1.0, 0.0,                0.0],
      [-np.sin(delta_rad_Y), 0.0, np.cos(delta_rad_Y), 0.0],
      [0.0,                0.0, 0.0,                1.0],
    ])
    M = rotation_matrix @ M
  

/*
 * CUDA Kernel for BarycentricWarp - Parallel triangle rasterization
 *
 * Massively parallel GPU rasterization for mesh-based warping.
 * Expected speedup: 10-20× over CPU cv2.warpAffine implementation.
 *
 * Features:
 * - One thread block per triangle for parallel rasterization
 * - Barycentric coordinate interpolation
 * - Atomic blending for overlapping triangles
 * - Texture sampling for source image
 *
 * Author: AI-assisted CUDA optimization
 * License: MIT
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Texture for source image (bilinear/bicubic sampling)
texture<float4, cudaTextureType2D, cudaReadModeElementType> texSrcMesh;

/**
 * Compute barycentric coordinates for point (px, py) in triangle (v0, v1, v2)
 *
 * Returns (alpha, beta, gamma) where:
 *   P = alpha*v0 + beta*v1 + gamma*v2
 *   alpha + beta + gamma = 1
 *
 * If point is outside triangle, returns invalid coords (alpha < 0 or beta < 0 or gamma < 0)
 */
__device__ float3 compute_barycentric(
    float px, float py,
    float2 v0, float2 v1, float2 v2
) {
    float denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);

    if (fabsf(denom) < 1e-6f) {
        // Degenerate triangle
        return make_float3(-1.0f, -1.0f, -1.0f);
    }

    float alpha = ((v1.y - v2.y) * (px - v2.x) + (v2.x - v1.x) * (py - v2.y)) / denom;
    float beta = ((v2.y - v0.y) * (px - v2.x) + (v0.x - v2.x) * (py - v2.y)) / denom;
    float gamma = 1.0f - alpha - beta;

    return make_float3(alpha, beta, gamma);
}

/**
 * CUDA Kernel: Rasterize a batch of triangles
 *
 * Each thread block handles one triangle, with threads rasterizing pixels inside the triangle's bounding box.
 *
 * @param dst_vertices  Deformed triangle vertices [num_triangles, 3, 2] (output space)
 * @param src_vertices  Source triangle vertices [num_triangles, 3, 2] (UV space, pixel coords)
 * @param output        Output image [H, W, C] (accumulated)
 * @param coverage      Coverage mask [H, W] (for blending overlapping triangles)
 * @param img_w         Output image width
 * @param img_h         Output image height
 * @param num_triangles Total number of triangles
 */
__global__ void rasterize_triangles_kernel(
    const float* dst_vertices,  // [num_tri, 3, 2]
    const float* src_vertices,  // [num_tri, 3, 2]
    float* output,
    float* coverage,
    int img_w,
    int img_h,
    int num_triangles
) {
    // Block ID = triangle index
    int tri_idx = blockIdx.x;
    if (tri_idx >= num_triangles) return;

    // Load triangle vertices
    int base_idx = tri_idx * 6;  // 3 vertices × 2 coords
    float2 dst_v0 = make_float2(dst_vertices[base_idx + 0], dst_vertices[base_idx + 1]);
    float2 dst_v1 = make_float2(dst_vertices[base_idx + 2], dst_vertices[base_idx + 3]);
    float2 dst_v2 = make_float2(dst_vertices[base_idx + 4], dst_vertices[base_idx + 5]);

    float2 src_v0 = make_float2(src_vertices[base_idx + 0], src_vertices[base_idx + 1]);
    float2 src_v1 = make_float2(src_vertices[base_idx + 2], src_vertices[base_idx + 3]);
    float2 src_v2 = make_float2(src_vertices[base_idx + 4], src_vertices[base_idx + 5]);

    // Compute bounding box
    float x_min = fmaxf(0.0f, floorf(fminf(fminf(dst_v0.x, dst_v1.x), dst_v2.x)));
    float x_max = fminf((float)img_w, ceilf(fmaxf(fmaxf(dst_v0.x, dst_v1.x), dst_v2.x)));
    float y_min = fmaxf(0.0f, floorf(fminf(fminf(dst_v0.y, dst_v1.y), dst_v2.y)));
    float y_max = fminf((float)img_h, ceilf(fmaxf(fmaxf(dst_v0.y, dst_v1.y), dst_v2.y)));

    int bbox_w = (int)(x_max - x_min);
    int bbox_h = (int)(y_max - y_min);

    if (bbox_w <= 0 || bbox_h <= 0) return;

    // Thread indices map to pixels in bounding box
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Stride loop to handle large bounding boxes
    for (int dy = local_y; dy < bbox_h; dy += blockDim.y) {
        for (int dx = local_x; dx < bbox_w; dx += blockDim.x) {
            float px = x_min + dx + 0.5f;  // Pixel center
            float py = y_min + dy + 0.5f;

            // Check if pixel is inside triangle (barycentric test)
            float3 bary = compute_barycentric(px, py, dst_v0, dst_v1, dst_v2);

            if (bary.x >= 0.0f && bary.y >= 0.0f && bary.z >= 0.0f) {
                // Inside triangle - interpolate source coordinates
                float src_x = bary.x * src_v0.x + bary.y * src_v1.x + bary.z * src_v2.x;
                float src_y = bary.x * src_v0.y + bary.y * src_v1.y + bary.z * src_v2.y;

                // Sample source image (bilinear via texture)
                float4 color = tex2D(texSrcMesh, src_x + 0.5f, src_y + 0.5f);

                // Write to output with atomic blend (handle overlapping triangles)
                int out_x = (int)px;
                int out_y = (int)py;

                if (out_x >= 0 && out_x < img_w && out_y >= 0 && out_y < img_h) {
                    int out_idx = (out_y * img_w + out_x) * 4;

                    atomicAdd(&output[out_idx + 0], color.x);
                    atomicAdd(&output[out_idx + 1], color.y);
                    atomicAdd(&output[out_idx + 2], color.z);
                    atomicAdd(&output[out_idx + 3], color.w);
                    atomicAdd(&coverage[out_y * img_w + out_x], 1.0f);
                }
            }
        }
    }
}

/**
 * CUDA Kernel: Normalize output by coverage (average overlapping triangles)
 */
__global__ void normalize_mesh_output_kernel(
    float* output,
    const float* coverage,
    int img_w,
    int img_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h) return;

    int idx = y * img_w + x;
    float cov = coverage[idx];

    if (cov > 0.5f) {  // At least one triangle covered this pixel
        int out_idx = idx * 4;
        output[out_idx + 0] /= cov;
        output[out_idx + 1] /= cov;
        output[out_idx + 2] /= cov;
        output[out_idx + 3] /= cov;
    }
}

// ------------------------------------------------------------
// Host API functions
// ------------------------------------------------------------

extern "C" {

/**
 * Bind source image to texture
 */
void bind_mesh_source_texture(const float* h_image, int img_w, int img_h, int channels) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, img_w, img_h);

    // Convert to RGBA
    float4* h_image_rgba = new float4[img_w * img_h];
    for (int i = 0; i < img_w * img_h; i++) {
        h_image_rgba[i].x = h_image[i * channels + 0];
        h_image_rgba[i].y = (channels > 1) ? h_image[i * channels + 1] : 0.0f;
        h_image_rgba[i].z = (channels > 2) ? h_image[i * channels + 2] : 0.0f;
        h_image_rgba[i].w = (channels > 3) ? h_image[i * channels + 3] : 1.0f;
    }

    cudaMemcpyToArray(cuArray, 0, 0, h_image_rgba, img_w * img_h * sizeof(float4), cudaMemcpyHostToDevice);
    delete[] h_image_rgba;

    texSrcMesh.addressMode[0] = cudaAddressModeClamp;
    texSrcMesh.addressMode[1] = cudaAddressModeClamp;
    texSrcMesh.filterMode = cudaFilterModeLinear;
    texSrcMesh.normalized = false;

    cudaBindTextureToArray(texSrcMesh, cuArray, channelDesc);
}

/**
 * Rasterize mesh (all triangles)
 */
void rasterize_mesh_cuda(
    const float* h_dst_vertices,
    const float* h_src_vertices,
    float* d_output,
    float* d_coverage,
    int img_w,
    int img_h,
    int num_triangles
) {
    // Allocate device memory for vertices
    float *d_dst_verts, *d_src_verts;
    size_t vert_size = num_triangles * 6 * sizeof(float);  // 3 verts × 2 coords
    cudaMalloc(&d_dst_verts, vert_size);
    cudaMalloc(&d_src_verts, vert_size);

    cudaMemcpy(d_dst_verts, h_dst_vertices, vert_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_verts, h_src_vertices, vert_size, cudaMemcpyHostToDevice);

    // Launch kernel: 1 block per triangle, 16×16 threads per block
    dim3 block(16, 16);
    dim3 grid(num_triangles);

    rasterize_triangles_kernel<<<grid, block>>>(
        d_dst_verts, d_src_verts, d_output, d_coverage,
        img_w, img_h, num_triangles
    );

    cudaDeviceSynchronize();

    cudaFree(d_dst_verts);
    cudaFree(d_src_verts);
}

/**
 * Normalize mesh output
 */
void normalize_mesh_output_cuda(float* d_output, float* d_coverage, int img_w, int img_h) {
    dim3 block(16, 16);
    dim3 grid((img_w + block.x - 1) / block.x, (img_h + block.y - 1) / block.y);

    normalize_mesh_output_kernel<<<grid, block>>>(d_output, d_coverage, img_w, img_h);
    cudaDeviceSynchronize();
}

} // extern "C"

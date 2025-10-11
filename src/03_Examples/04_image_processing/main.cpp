#include "../../include/simd_utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

/**
 * This example demonstrates using SIMD for basic image processing operations.
 * 
 * We'll implement:
 * 1. Brightness adjustment
 * 2. Contrast enhancement
 * 3. Image blurring (simple box filter)
 * 4. Grayscale conversion
 * 
 * For simplicity, we'll use a simulated image represented as a 1D array of pixels,
 * where each pixel has R, G, B components (3 bytes per pixel).
 */

// Simulated image dimensions (kept modest so benchmarks finish quickly)
const int WIDTH = 512;
const int HEIGHT = 384;
const int CHANNELS = 3;  // RGB
const int IMAGE_SIZE = WIDTH * HEIGHT * CHANNELS;

// Utility function to initialize a test image
void initialize_test_image(uint8_t* image, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            
            // Create a gradient pattern
            image[idx + 0] = static_cast<uint8_t>(x * 255 / width);  // R
            image[idx + 1] = static_cast<uint8_t>(y * 255 / height); // G
            image[idx + 2] = static_cast<uint8_t>(128);              // B
        }
    }
}

// Print a small section of the image for verification
void print_image_section(const uint8_t* image, int width, int channels, 
                         int start_x, int start_y, int section_width, int section_height) {
    std::cout << "Image section (" << start_x << "," << start_y << ") to (" 
              << start_x + section_width - 1 << "," << start_y + section_height - 1 << "):" << std::endl;
    
    for (int y = start_y; y < start_y + section_height; y++) {
        for (int x = start_x; x < start_x + section_width; x++) {
            int idx = (y * width + x) * channels;
            std::cout << "(" << static_cast<int>(image[idx + 0]) << ","
                      << static_cast<int>(image[idx + 1]) << ","
                      << static_cast<int>(image[idx + 2]) << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 1. Brightness adjustment - Scalar implementation
void adjust_brightness_scalar(uint8_t* image, int size, int brightness) {
    for (int i = 0; i < size; i++) {
        int value = static_cast<int>(image[i]) + brightness;
        image[i] = static_cast<uint8_t>(std::min(255, std::max(0, value)));
    }
}

// 1. Brightness adjustment - SIMD implementation
void adjust_brightness_simd(uint8_t* image, int size, int brightness) {
    // Create a vector with the brightness value
    __m256i brightness_vec = _mm256_set1_epi8(static_cast<char>(brightness));
    __m256i zero_vec = _mm256_setzero_si256();
    __m256i max_vec = _mm256_set1_epi8(static_cast<char>(255));
    
    // Process 32 bytes at a time (32 pixels)
    int i = 0;
    for (; i <= size - 32; i += 32) {
        // Load 32 bytes
        __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&image[i]));
        
        // Add brightness
        __m256i result = _mm256_adds_epu8(pixels, brightness_vec);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&image[i]), result);
    }
    
    // Handle remaining pixels
    for (; i < size; i++) {
        int value = static_cast<int>(image[i]) + brightness;
        image[i] = static_cast<uint8_t>(std::min(255, std::max(0, value)));
    }
}

// 2. Contrast enhancement - Scalar implementation
void enhance_contrast_scalar(uint8_t* image, int size, float contrast) {
    // Apply contrast formula: (pixel - 128) * contrast + 128
    for (int i = 0; i < size; i++) {
        float value = (static_cast<float>(image[i]) - 128.0f) * contrast + 128.0f;
        image[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, value)));
    }
}

// 2. Contrast enhancement - SIMD implementation
void enhance_contrast_simd(uint8_t* image, int size, float contrast) {
    // We'll process 8 pixels at a time (converting to float for the calculation)
    __m256 contrast_vec = _mm256_set1_ps(contrast);
    __m256 offset_vec = _mm256_set1_ps(128.0f);
    __m256 min_vec = _mm256_setzero_ps();
    __m256 max_vec = _mm256_set1_ps(255.0f);
    
    // Process 8 pixels at a time
    int i = 0;
    for (; i <= size - 8; i += 8) {
        // Load 8 bytes and convert to float
        __m128i pixels_epi8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&image[i]));
        __m256i pixels_epi32 = _mm256_cvtepu8_epi32(pixels_epi8);
        __m256 pixels_ps = _mm256_cvtepi32_ps(pixels_epi32);
        
        // Apply contrast formula: (pixel - 128) * contrast + 128
        __m256 centered = _mm256_sub_ps(pixels_ps, offset_vec);
        __m256 scaled = _mm256_mul_ps(centered, contrast_vec);
        __m256 result_ps = _mm256_add_ps(scaled, offset_vec);
        
        // Clamp to [0, 255]
        result_ps = _mm256_min_ps(_mm256_max_ps(result_ps, min_vec), max_vec);
        
        // Convert back to integers and store without requiring AVX-512
        __m256i result_epi32 = _mm256_cvtps_epi32(result_ps);
        __m128i result_low = _mm256_castsi256_si128(result_epi32);
        __m128i result_high = _mm256_extracti128_si256(result_epi32, 1);
        __m128i packed16 = _mm_packus_epi32(result_low, result_high);
        __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&image[i]), packed8);
    }
    
    // Handle remaining pixels
    for (; i < size; i++) {
        float value = (static_cast<float>(image[i]) - 128.0f) * contrast + 128.0f;
        image[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, value)));
    }
}

// 3. Grayscale conversion - Scalar implementation
void convert_to_grayscale_scalar(const uint8_t* src, uint8_t* dst, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * CHANNELS;
            int dst_idx = y * width + x;
            
            // Standard grayscale conversion weights
            uint8_t gray = static_cast<uint8_t>(
                0.299f * src[src_idx + 0] +  // R
                0.587f * src[src_idx + 1] +  // G
                0.114f * src[src_idx + 2]    // B
            );
            
            dst[dst_idx] = gray;
        }
    }
}

// 3. Grayscale conversion - SIMD implementation
void convert_to_grayscale_simd(const uint8_t* src, uint8_t* dst, int width, int height) {
	// RGB to Grayscale conversion weights
	const float weight_r = 0.299f;
	const float weight_g = 0.587f;
	const float weight_b = 0.114f;

	const __m128 weight_r_vec = _mm_set1_ps(weight_r);
	const __m128 weight_g_vec = _mm_set1_ps(weight_g);
	const __m128 weight_b_vec = _mm_set1_ps(weight_b);
	const __m128i r_shuffle = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	const __m128i g_shuffle = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	const __m128i b_shuffle = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	const __m128i zero_128 = _mm_setzero_si128();
	alignas(16) uint8_t chunk[16];

	const int row_stride = width * CHANNELS;
	for (int y = 0; y < height; y++) {
		const uint8_t* row_ptr = src + y * row_stride;
		uint8_t* dst_row = dst + y * width;
		int x = 0;
		for (; x <= width - 4; x += 4) {
			const uint8_t* pixel_ptr = row_ptr + x * CHANNELS;
			std::memcpy(chunk, pixel_ptr, 12);
			__m128i block = _mm_load_si128(reinterpret_cast<const __m128i*>(chunk));

			__m128i r_bytes = _mm_shuffle_epi8(block, r_shuffle);
			__m128i g_bytes = _mm_shuffle_epi8(block, g_shuffle);
			__m128i b_bytes = _mm_shuffle_epi8(block, b_shuffle);

			__m128 r_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(r_bytes));
			__m128 g_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(g_bytes));
			__m128 b_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b_bytes));

			__m128 gray_ps = _mm_mul_ps(r_ps, weight_r_vec);
			gray_ps = _mm_add_ps(gray_ps, _mm_mul_ps(g_ps, weight_g_vec));
			gray_ps = _mm_add_ps(gray_ps, _mm_mul_ps(b_ps, weight_b_vec));

			__m128i gray_epi32 = _mm_cvtps_epi32(gray_ps);
			__m128i gray_epi16 = _mm_packus_epi32(gray_epi32, zero_128);
			__m128i gray_epi8 = _mm_packus_epi16(gray_epi16, zero_128);

			int packed = _mm_cvtsi128_si32(gray_epi8);
			std::memcpy(dst_row + x, &packed, sizeof(packed));
		}
		for (; x < width; x++) {
			int src_idx = (y * width + x) * CHANNELS;
			float r = static_cast<float>(src[src_idx + 0]);
			float g = static_cast<float>(src[src_idx + 1]);
			float b = static_cast<float>(src[src_idx + 2]);
			float gray = r * weight_r + g * weight_g + b * weight_b;
			dst_row[x] = static_cast<uint8_t>(gray);
		}
	}
}

int main() {
    set_benchmark_suite("03_Examples/04_image_processing");

    std::cout << "=== SIMD Image Processing Example ===" << std::endl;
    
    // Allocate memory for the test image
	uint8_t* original_image = new uint8_t[IMAGE_SIZE + 32];
	uint8_t* processed_image = new uint8_t[IMAGE_SIZE + 32];
	uint8_t* grayscale_image = new uint8_t[WIDTH * HEIGHT + 32];
    
    // Initialize the test image
    initialize_test_image(original_image, WIDTH, HEIGHT, CHANNELS);
    
    // Print a small section of the original image
    std::cout << "Original Image:" << std::endl;
    print_image_section(original_image, WIDTH, CHANNELS, 0, 0, 3, 3);
    
    // 1. Brightness Adjustment
    std::cout << "1. Brightness Adjustment" << std::endl;
    
    // Copy original image to processed image
    std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
    
    // Benchmark brightness adjustment
	auto brightness_scalar = [&]() {
		std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
		adjust_brightness_scalar(processed_image, IMAGE_SIZE, 50);
	};

	auto brightness_simd = [&]() {
		std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
		adjust_brightness_simd(processed_image, IMAGE_SIZE, 50);
	};
    
    benchmark_comparison("Brightness Adjustment", brightness_scalar, brightness_simd, 10);
    
    // Print a small section of the brightness-adjusted image
    std::cout << "Brightness-adjusted Image:" << std::endl;
    print_image_section(processed_image, WIDTH, CHANNELS, 0, 0, 3, 3);
    
    // 2. Contrast Enhancement
    std::cout << "2. Contrast Enhancement" << std::endl;
    
    // Reset the processed image
    std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
    
    // Benchmark contrast enhancement
	auto contrast_scalar = [&]() {
		std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
		enhance_contrast_scalar(processed_image, IMAGE_SIZE, 1.5f);
	};

	auto contrast_simd = [&]() {
		std::copy(original_image, original_image + IMAGE_SIZE, processed_image);
		enhance_contrast_simd(processed_image, IMAGE_SIZE, 1.5f);
	};
    
    benchmark_comparison("Contrast Enhancement", contrast_scalar, contrast_simd, 10);
    
    // Print a small section of the contrast-enhanced image
    std::cout << "Contrast-enhanced Image:" << std::endl;
    print_image_section(processed_image, WIDTH, CHANNELS, 0, 0, 3, 3);
    
    // 3. Grayscale Conversion
    std::cout << "3. Grayscale Conversion" << std::endl;
    
    // Benchmark grayscale conversion
    auto grayscale_scalar = [&]() {
        convert_to_grayscale_scalar(original_image, grayscale_image, WIDTH, HEIGHT);
    };
    
    auto grayscale_simd = [&]() {
        convert_to_grayscale_simd(original_image, grayscale_image, WIDTH, HEIGHT);
    };
    
    benchmark_comparison("Grayscale Conversion", grayscale_scalar, grayscale_simd, 10);
    
    // Print a small section of the grayscale image
    std::cout << "Grayscale Image (showing first few pixels):" << std::endl;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            std::cout << static_cast<int>(grayscale_image[y * WIDTH + x]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] original_image;
    delete[] processed_image;
    delete[] grayscale_image;
    
    return 0;
} 

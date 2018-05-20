// HpcImageRotation.cpp : Defines the entry point for the console application.

#include "stdafx.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <exception>
#include "tga/tga.h"

#pragma region global
cl::Platform platform;
cl::Program program;
cl::Context context;
cl_int err = CL_SUCCESS;
std::vector<cl::Device> devices;
#pragma endregion

#pragma region OPENCL_HELPERS
void print_OpenCl_Error(cl::Error err);
bool setup_OpenCl_Platform();
void print_OpenCl_Platform(cl::Platform platform);
#pragma endregion

#pragma region CONSTANTS
const std::string KERNEL_FILE = "rotation.cl";
const std::string IMAGE_FILE = "lenna.tga";
const std::string IMAGE_ROTATED_FILE = "lenna_rotated.tga";
const bool DEBUG = false;
const bool PRINT = false;
#pragma endregion

#pragma region METHODS
float degreeToRadians(int degree);
int getDegrees(int argc, char * argv[]);
tga::TGAImage * loadImage(const std::string filename);
bool rotateImage(tga::TGAImage * image, float theta);
tga::TGAImage * createTGAImage(tga::TGAImage * original, std::vector<int> imageData);
void writeImage(std::string file, tga::TGAImage * image);
#pragma endregion

int  main(int argc, char * argv[])
{
	return setup_OpenCl_Platform() ? rotateImage(loadImage(IMAGE_FILE), degreeToRadians(getDegrees(argc, argv))) : EXIT_FAILURE;
}

bool rotateImage(tga::TGAImage * tga_image, float theta) {
	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);

		const int size = tga_image->imageData.size();

		std::vector<int> src_rgb;
		for (int i = 0; i < size; i++) {
			src_rgb.push_back((int)tga_image->imageData[i]);
		}
		std::vector<int> dest_rgb(size);

		std::cout << "Prepare buffers ..." << std::endl;
		cl::Buffer source_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size * sizeof(cl_int));
		cl::Buffer dest_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, size * sizeof(cl_int));

		std::cout << "Write to device ..." << std::endl;
		queue.enqueueWriteBuffer(
			source_buffer,
			CL_TRUE,
			0,
			size * sizeof(cl_int),
			&src_rgb[0]
		);

		cl::Kernel addKernel(program, "rotate_image", &err);

		addKernel.setArg(0, source_buffer);
		addKernel.setArg(1, dest_buffer);
		addKernel.setArg(2, tga_image->height);
		addKernel.setArg(3, tga_image->width);
		addKernel.setArg(4, sinf(theta));
		addKernel.setArg(5, cosf(theta));
		// RGB type ? uchar3 : --> RGBA -> uchar4 (RGB + alpha channel, lenna = uchar3)
		addKernel.setArg(6, (tga_image->type == 1 ? 4 : 3));

		cl::NDRange offset(0);
		cl::NDRange global(tga_image->width, tga_image->height, tga_image->type == 1 ? 4 : 3);
		cl::NDRange local(1, 1, 1);

		queue.enqueueNDRangeKernel(addKernel, offset, global, local);

		cl::Event event;
		queue.enqueueReadBuffer(
			dest_buffer,
			CL_TRUE,
			0,
			size * sizeof(cl_int),
			&dest_rgb[0],
			NULL,
			&event
		);

		queue.finish();
		event.wait();

		std::cout << "Read buffer from device ..." << std::endl;
		writeImage(IMAGE_ROTATED_FILE, createTGAImage(tga_image, dest_rgb));
		
		src_rgb.clear();
		dest_rgb.clear();
		delete tga_image;
		return EXIT_SUCCESS;
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
		return EXIT_FAILURE;
	}
}

tga::TGAImage * createTGAImage(tga::TGAImage * original, std::vector<int> imageData) {
	tga::TGAImage * tga_rotated = new tga::TGAImage();
	tga_rotated->bpp = original->bpp;
	tga_rotated->height = original->height;
	tga_rotated->width = original->width;
	tga_rotated->type = original->type;
	try {
		tga_rotated->imageData.clear();
		for (unsigned int i = 0; i < imageData.size(); i++) {
			tga_rotated->imageData.push_back((unsigned char)imageData[i]);
		}
	}
	catch (int e) {
		std::cout << "Error: " << e << std::endl;
	}
	return tga_rotated;
}

void writeImage(std::string file, tga::TGAImage * image) {
	tga::saveTGA(*image, file.c_str());
	std::cout << "Image written to disk" << std::endl;
}

tga::TGAImage * loadImage(const std::string filename) {
	tga::TGAImage * tga_image = new tga::TGAImage();
	if (tga::LoadTGA(tga_image, filename.c_str())) {
		std::cout << "Image loaded" << std::endl;
	}
	else {
		std::cout << "Image could not be loaded!" << std::endl;
	}
	return tga_image;
}

int getDegrees(int argc, char * argv[]) {
	std::cout << "******************************************" << std::endl;
	int degrees = 0;
	if (argc == 2) {
		degrees = strtol(argv[1], nullptr, 0);
	}
	else {
		std::cout << "Please enter the degrees for rotation: " << std::endl;
		std::cin >> degrees;
	}
	return degrees;
}

float degreeToRadians(int degrees) {
	float theta = (degrees * CL_M_PI) / 180.0f;
	std::cout << "Rotating by: " << degrees << " degrees (radians: " << theta << ")" << std::endl;
	std::cout << "******************************************" << std::endl;
	return theta;
}

bool setup_OpenCl_Platform()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cout << "No OpenCL platforms available!\n";
		return false;
	}
	platform = platforms.size() == 2 ? platforms[1] : platforms[0];
	if (DEBUG)
		print_OpenCl_Platform(platform);
	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	std::ifstream sourceFile(KERNEL_FILE);
	if (!sourceFile)
	{
		std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
		return false;
	}
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	program = cl::Program(context, source);
	program.build(devices);
	return true;
}

void print_OpenCl_Platform(cl::Platform platform)
{
	const cl_platform_info attributeTypes[5] = {
		CL_PLATFORM_NAME,
		CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION,
		CL_PLATFORM_PROFILE,
		CL_PLATFORM_EXTENSIONS };

	std::cout << "**********************************************" << std::endl;
	std::cout << "Selected Platform Information: " << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
	std::cout << "**********************************************" << std::endl;
}

void print_OpenCl_Error(cl::Error err)
{
	std::string s;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &s);
	std::cout << s << std::endl;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_OPTIONS, &s);
	std::cout << s << std::endl;

	std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
}
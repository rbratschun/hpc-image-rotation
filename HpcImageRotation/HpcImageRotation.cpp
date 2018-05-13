// HpcImageRotation.cpp : Defines the entry point for the console application.

#include "stdafx.h"


// NVidia only supports OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <exception>
#include "tga/tga.h"

const int SELECTED_PLATFORM = 1;

int  main(void)
{

	const std::string KERNEL_FILE = "rotation.cl";
	cl_int err = CL_SUCCESS;
	cl::Program program;
	std::vector<cl::Device> devices;

	try {

		// get available platforms ( NVIDIA, Intel, AMD,...)
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "No OpenCL platforms available!\n";
			return EXIT_FAILURE;
		}
		cl::Platform platform;
		if (platforms.size() == 1)
			platform = platforms[0];
		else
			platform = platforms[1];

		// create a context and get available devices
		const cl_platform_info attributeTypes[5] = {
			CL_PLATFORM_NAME,
			CL_PLATFORM_VENDOR,
			CL_PLATFORM_VERSION,
			CL_PLATFORM_PROFILE,
			CL_PLATFORM_EXTENSIONS };

		// cl::Platform platform = platforms[SELECTED_PLATFORM]; 
		// on a different machine, you may have to select a different platform!

		std::cout << "Selected Platform Information: " << std::endl;
		std::cout << "Platform ID: " << SELECTED_PLATFORM << std::endl;
		std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
		std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		cl_context_properties properties[] =
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[SELECTED_PLATFORM])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);

		devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// load and build the kernel
		std::ifstream sourceFile(KERNEL_FILE);
		if (!sourceFile)
		{
			std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
			return 1;
		}
		std::string sourceCode(
			std::istreambuf_iterator<char>(sourceFile),
			(std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
		program = cl::Program(context, source);
		program.build(devices);

		//create kernels
		cl::Event event;

		// create command queue
		cl::CommandQueue queue(context, devices[0], 0, &err);

		tga::TGAImage * tga_image = new tga::TGAImage();
		std::string file_name = "lenna.tga";

		if (tga::LoadTGA(tga_image, file_name.c_str())) {
			std::cout << "Image loaded successfully!" << std::endl;
		}
		else {
			std::cout << "Image could not be loaded!" << std::endl;
			return EXIT_FAILURE;
		}

		const int size = tga_image->imageData.size();

		std::vector<int> src_rgb;
		for (int i = 0; i<size; i++) {
			src_rgb.push_back((int)tga_image->imageData[i]);
		}
		std::vector<int> dest_rgb;
		dest_rgb.resize(src_rgb.size());

		// INPUT BUFFER = tga_image->imageData (1D array)
		cl::Buffer source_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size * sizeof(int));
		// OUTPUT BUFFER
		cl::Buffer dest_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, size * sizeof(int));

		if (tga_image->type == 0) {
			std::cout << "image type: " << "RGB (uchar3)" << std::endl;
		}
		else if (tga_image->type == 1) {
			std::cout << "image type: " << "RGBA (uchar4)" << std::endl;
		}
		else {
			std::cout << "image type: " << "unknown" << std::endl;
			return EXIT_FAILURE;
		}

		float theta = 3.14159f / 2.0f; // *3 / 2.0f; // THETA in radians (odd * pi = vertical, even * pi horizontal rotation)
		float cos_theta = cosf(theta);
		float sin_theta = sinf(theta);


		// WRITE IMAGEDATA TO GPU BUFFER
		queue.enqueueWriteBuffer(
			source_buffer, // SOURCE BUFFER
			CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
			0, // OFFSET
			size * sizeof(int), // SIZE OF ARRAY 
			&src_rgb[0] // POINTER TO VECTOR
		);

		// CREATE KERNEL
		cl::Kernel addKernel(program, "rotate_image", &err);

		// SET KERNEL ARGUMENTS
		addKernel.setArg(0, source_buffer);
		addKernel.setArg(1, dest_buffer);
		addKernel.setArg(2, tga_image->height);
		addKernel.setArg(3, tga_image->width);
		addKernel.setArg(4, sin_theta);
		addKernel.setArg(5, cos_theta);
		// RGB type ? uchar3 : --> RGBA -> uchar4 (RGB + alpha channel, lenna = uchar3)
		addKernel.setArg(6, (tga_image->type == 1 ? 4 : 3));

		// PREPARE ADD KERNEL
		// SET ND RANGE FOR KERNEL (matrix size, local and global ranges)
		cl::NDRange global(tga_image->width, tga_image->height, tga_image->type == 1 ? 4 : 3);
		cl::NDRange local(1,1,1); //make sure local range is divisible by global range
		cl::NDRange offset(0); // START FROM 0,0
		std::cout << "CALL 'rotate_image' KERNEL" << std::endl;
		// ENQUEUE KERNEL EXECUTION
		queue.enqueueNDRangeKernel(addKernel, offset, global, local);

		// PREPARE AND WAIT FOR KERNEL RESULT
		queue.enqueueReadBuffer(
			dest_buffer,
			CL_TRUE,
			0,
			size * sizeof(int),
			&dest_rgb[0],
			NULL,
			&event
		);

		// WAITING FOR KERNEL FINISH EVENT
		event.wait();
		std::cout << "Received data from kernel" << std::endl;

		// CREATE NEW TGA IMAGE FROM DEST_BUFFER
		tga::TGAImage * tga_rotated = new tga::TGAImage();
		tga_rotated->bpp = tga_image->bpp;
		tga_rotated->height = tga_image->height;
		tga_rotated->width = tga_image->width;
		tga_rotated->type = tga_image->type;
		try {
			tga_rotated->imageData.clear();
			for (unsigned int i = 0; i < dest_rgb.size(); i++) {
				tga_rotated->imageData.push_back((unsigned char)dest_rgb[i]);
			}
			// WRITE ROTATED IMAGE TO DISK
			std::string rotated_image_file = "lenna_rotated.tga";
			tga::saveTGA(*tga_rotated, rotated_image_file.c_str());
			std::cout << "image file written successfully" << std::endl;
		}
		catch (int e) {
			// PROBLEM WRITING IMAGE DATA / FILE
			std::cout << "Error: " << e << std::endl;
		}
		// CLEAN UP
		src_rgb.clear();
		dest_rgb.clear();

		return EXIT_SUCCESS;
	}
	catch (cl::Error err) {
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
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;

}

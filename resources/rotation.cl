__kernel 
void rotate_image(
	__global const int * restrict src_buffer,
	__global int * dest_buffer,
	const int width,
	const int height,
	const float sin,
	const float cos,
	const int rgb_factor)
	{
	
	// ROTATION AROUND CENTER COORDS
    const float x0 = (float) width / 2.0f;
    const float y0 = (float) height / 2.0f;
	
	// X / Y AS GLOBAL IDX (FOR)
    const int x1 = get_global_id(0);
    const int y1 = get_global_id(1);
	const int channel = get_global_id(2);

	// CALCULATE NEW (ROTATED) POSITION FOR CURRENT COORDINATES
	float x2 = cos * ( x1 - x0 ) - sin * ( y1 - y0 ) + x0;
	float y2 = sin * ( x1 - x0 ) + cos * ( y1 - y0 ) + y0;
	
	// TRANSLATE COORDINATES TO 1D RGB/A VECTOR
	int pixelSource  = floor( floor(x2) + (floor(y2) * width)) * rgb_factor;
	int pixelDest = (x1 + y1 * width) * rgb_factor;
	
	// WRITE RGB VALUES TO DESTINATION BUFFER
	dest_buffer[pixelDest + channel] = src_buffer[pixelSource + channel];
}
	
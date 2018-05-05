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
    const float x0 = (int) width / 2.0f;
    const float y0 = (int) height / 2.0f;
	
	// X / Y AS GLOBAL IDX (FOR)
    const int x1 = get_global_id(0);
    const int y1 = get_global_id(1);

	// CALCULATE NEW (ROTATED) POSITION FOR CURRENT COORDINATES
	float x2 = cos * ((float) x1 - x0 ) - sin * ((float) y1 - y0 ) + x0;
	float y2 = sin * ((float) x1 - x0 ) + cos * ((float) y1 - y0 ) + y0;
	
	// TRANSLATE COORDINATES TO 1D RGB/A VECTOR
	int pixelSource  = (int) floor((floor(y2) * width + floor(x2))) * rgb_factor;
	int pixelDest = (x1 + y1 * width) * rgb_factor;
	
	// WRITE RGB VALUES TO DESTINATION BUFFER
	dest_buffer[pixelDest + 0] = src_buffer[pixelSource + 0];
	dest_buffer[pixelDest + 1] = src_buffer[pixelSource + 1];
	dest_buffer[pixelDest + 2] = src_buffer[pixelSource + 2];
}
	
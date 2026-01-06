#include <sys/time.h>
#include <cstdlib>
#include "timing.h"


long getTimeMilliseconds()
{
	struct timeval current;
	
	gettimeofday(&current, NULL);
	
	return (current.tv_sec * 1000) + (current.tv_usec / 1000);
}



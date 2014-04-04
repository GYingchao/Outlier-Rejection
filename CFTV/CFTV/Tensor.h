#include <ANN/ANN.h>					// ANN declarations
class Tensor
{
public:

	// a 3x3 matrix
	double a[3][3];
	// data point which it stands for
	ANNpoint queryPt;

public:
	Tensor(ANNpoint pt) 
	{	
		// 
	}

	~Tensor(){}
};
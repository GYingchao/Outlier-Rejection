#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <ANN/ANN.h>					// ANN declarations
using namespace std;

ANNpointArray dataPts;					// data points
ANNkd_tree*	kdTree;						// search structure
int ptNums;								// number of data points
vector<double> x_array;
vector<double> y_array;
vector<double> z_array;
vector<int> d_tag;						// store the 4th coordinate of the data points in viewworld format
double sigma = 5.0;

/*
static double** matrixMultiplication(double A[n][n], double B[n][n], int n) 
{
	double** c = new double*[n];
	for(int i=0; i<n; i++) c[i] = new double[n];

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			c[i][j] = 0;
			for(int k=0; k<n; k++) {
				c[i][j] += A[i][k]*B[k][j];
			}
		}
	}

	return c;
}
*/

int main () {
		
	// Load data set
	  string line;
	  ptNums = 0;
	  ifstream myfile("D:/YingchaoDoc2/codes/TV/input/cas.noise200.dat");
	  if (myfile.is_open()) {
		while (getline(myfile, line)) {
			//cout << line << '\n';
			istringstream in(line);
			double x, y, z;
			int d;
			in >> x >> y >> z >> d;
			//in >> dataPts[ptNums][0] >> dataPts[ptNums][1] >> dataPts[ptNums][2];
			//in >> d;
			//dataPts[ptNums][0] = x;
			x_array.push_back(x);
			y_array.push_back(y);
			z_array.push_back(z);
			d_tag.push_back(d);
			//cout << x << ", " << y << ", " << z << ", " << d << endl;
			ptNums ++;
		}
		myfile.close();
		cout << ptNums << " lines in total.." << endl;
	  } else cout << "Unable to open file" << endl; 

	// Use ANN data structure
	  dataPts = annAllocPts(ptNums, 3);			// allocate data points, dim=3 by default
	  
	  for(int i=0; i<ptNums; i++) {
		  dataPts[i][0] = x_array[i];
		  dataPts[i][1] = y_array[i];
		  dataPts[i][2] = z_array[i];
	  }
	  
	  /*
	  for(int i=0; i<ptNums; i++) {
		  cout << dataPts[i][0] << endl;
	  }
	  */

	// Build the ANN tree
	  kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					ptNums,						// number of points
					3);							// dimension of space


	// Test ANN tree
	  /*
	  int neighbors = kdTree->annkFRSearch(dataPts[1], 5, 0);
	  int* ptId = new int[neighbors];
	  double* ptDis = new double[neighbors];
	  kdTree->annkFRSearch(dataPts[1], 5, neighbors, ptId, ptDis);
	  for(int i=0; i<neighbors; i++) {
		  cout << ptId[i] << endl; 
	  }
	   for(int i=0; i<neighbors; i++) {
		   cout << ptDis[i] << endl;
	   }
	   */

	// Initialize the (ball)tensors
	  double*** tensors = new double**[ptNums];
	  for(int i=0; i<ptNums; i++) {
		  tensors[i] = new double*[3];
		  for(int j=0; j<3; j++) {
			  tensors[i][j] = new double[3];
			  for(int k=0; k<3; k++) {
				  // Identity Matrix
				  if(j==k) tensors[i][j][k] = 1.0;
				  else tensors[i][j][k] = 0.0;
			  }
		  }
	  }
	
	// One pass close form tensor voting
	  for(int i=0; i<ptNums; i++) {

		  // Start up by the first point
			int neighbors = kdTree->annkFRSearch(dataPts[i], sigma, 0);
		  // Get the neighbors of the qury point
			int* ptId = new int[neighbors];
			double* ptDis = new double[neighbors];
			kdTree->annkFRSearch(dataPts[i], sigma, neighbors, ptId, ptDis);
			for(int n=0; n<neighbors; n++) {
				// process every neighbor by CFTV
				int j = ptId[n];
				// Now the voter is j and receiver is i

				// Compute r_ij
				double* r_ij = new double[3];
				double r_norm = 0;
				for(int p=0; p<3; p++) {
					r_ij[p] = dataPts[i][p] - dataPts[j][p];
					r_norm += r_ij[p]*r_ij[p];
				}
				for(int p=0; p<3; p++) {
					r_ij[p] /= sqrt(r_norm);
				}

				// Compute R_ij
				double R_ij[3][3] = {{1-2*r_ij[0]*r_ij[0], -2*r_ij[0]*r_ij[1], -2*r_ij[0]*r_ij[2]}, 
									 {-2*r_ij[1]*r_ij[0], 1-2*r_ij[1]*r_ij[1], -2*r_ij[1]*r_ij[2]},
									 {-2*r_ij[2]*r_ij[0], -2*r_ij[2]*r_ij[1], 1-2*r_ij[2]*r_ij[2]}};
				
				// Test matrix multiplication
				/*
				double A[3][3] = {{0, 3, 2}, {0.1, 0.7, 0.5}, {1.2, 3.4, 8.0}};
				double B[3][3] = {{5, 4, 3}, {2, 1, 0}, {9, 8, 7}};

				double** c = new double*[3];
				for(int i=0; i<3; i++) c[i] = new double[3];
				for(int i=0; i<3; i++) {
					for(int j=0; j<3; j++) {
						c[i][j] = 0;
						for(int k=0; k<3; k++) {
							c[i][j] += A[i][k]*B[k][j];
						}
					}
				}
				*/
				/*
				for(int m=0; m<3; m++) {
					for(int n=0; n<3; n++) {
						cout << c[m][n] << " ";
					}
					cout << endl;
				}
				*/

				// Compute R_ij_
				double tem[3][3] = {{1-0.5*r_ij[0]*r_ij[0], -0.5*r_ij[0]*r_ij[1], -0.5*r_ij[0]*r_ij[2]}, 
									 {-0.5*r_ij[1]*r_ij[0], 1-0.5*r_ij[1]*r_ij[1], -0.5*r_ij[1]*r_ij[2]},
									 {-0.5*r_ij[2]*r_ij[0], -0.5*r_ij[2]*r_ij[1], 1-0.5*r_ij[2]*r_ij[2]}};
				double** R_ij_ = new double*[3];
				for(int ii=0; ii<3; ii++) R_ij_[ii] = new double[3];
				for(int ii=0; ii<3; ii++) {
					for(int jj=0; jj<3; jj++) {
						R_ij_[ii][jj] = 0;
						for(int k=0; k<3; k++) {
							R_ij_[ii][jj] += tem[ii][k]*R_ij[k][jj];
						}
					}
				}

				// Compute c_ij
				double norminator = 0;
				for(int p=0; p<3; p++) {
					norminator += r_ij[p]*r_ij[p];
				}
				double c_ij = exp(-norminator/sigma);

				// Compute S_ij

				/// 1 Compute K_jR_ij_
				double** t1 = new double*[3];
				for(int k=0; k<3; k++) t1[k] = new double[3];
				for(int ii=0; ii<3; ii++) {
					for(int jj=0; jj<3; jj++) {
						t1[ii][jj] = 0;
						for(int k=0; k<3; k++) {
							t1[ii][jj] += tensors[j][ii][k]*R_ij_[k][jj];
						}
					}
				}

				/// 2 Compute R_ijK_jR_ij_
				double** t2 = new double*[3];
				for(int k=0; k<3; k++) t2[k] = new double[3];
				for(int ii=0; ii<3; ii++) {
					for(int jj=0; jj<3; jj++) {
						t2[ii][jj] = 0;
						for(int k=0; k<3; k++) {
							t2[ii][jj] += R_ij[ii][k]*t1[k][jj];
						}
					}
				}

				/// 3 Compute S_ij
				double** S_ij = new double*[3];
				for(int k=0; k<3; k++) S_ij[k] = new double[3];
				for(int ii=0; ii<3; ii++) {
					for(int jj=0; jj<3; jj++) {
						S_ij[ii][jj] = c_ij*t2[ii][jj];
					}
				}

				// Collect the votes for i
				for(int ii=0; ii<3; ii++) {
					for(int jj=0; jj<3; jj++) {
						tensors[i][ii][jj] += S_ij[ii][jj]; 
					}
				}
			}
	  } // One pass CFTV done
	


	  return 0;
}
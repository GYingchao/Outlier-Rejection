#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int main () {
  string line;
  int count = 0;
  ifstream myfile("D:/YingchaoDoc2/codes/TV/input/noisy_PS");
  if (myfile.is_open()) {
    while (getline(myfile, line)) {
		count ++;
		//cout << line << '\n';
		istringstream in(line);
		float x, y, z;
		int d;
		in >> x >> y >> z >> d;
		cout << x << ", " << y << ", " << z << ", " << d << endl;
    }
    myfile.close();
	cout << count << " lines in total.." << endl;
  }

  else cout << "Unable to open file" << endl; 

  return 0;
}
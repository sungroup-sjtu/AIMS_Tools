//#include "../../code_headfile/xy.h"
#include "/home/xiangyan/md1400/Github/MD_Analysis/XYHeadFile/xy.h"
using namespace std;

int main(int argc, char** argv)
{
	FILE* f_pre= fopen(argv[1], "r");
	double V = atof(argv[2]);
	double T = atof(argv[3]);
	double weight = atof(argv[4]);
	char weight_char[10];
	sprintf(weight_char, "%.2f", weight);
	vector <double> t;
	vector <double> pxy;
	vector <double> pxz;
	vector <double> pyz;
	char p[1000];
	string split_symbol = " \n\t";
	while (fgets(p, 1000, f_pre)) {
		vector <string> sp = split(string(p), split_symbol);
		if (sp.size() <= 1) {

		}
		else if (sp[1] == "s0" or sp[1] == "s1" or sp[1] == "s2") {
			if (sp[3] != "\"Pres-XY\"" and sp[3] != "\"Pres-XZ\"" and sp[3] != "\"Pres-YZ\"") {
				error("input file error\n");
			}
		}
		else if (p[0] != '#' and p[0] != '@') {
			t.push_back(atof(sp[0].c_str()));
			pxy.push_back(atof(sp[1].c_str()));
			pxz.push_back(atof(sp[2].c_str()));
			pyz.push_back(atof(sp[3].c_str()));
		}
	}
	fclose(f_pre);
	double dt = t[1] - t[0];
	vector <double> t_list;
	vector <double> acf_list;
	for (unsigned i = 0; i < (t.size() / 2); ++i) {
		//printf("\r%i / %i", i, (t.size() / 2));
		double Dt = i * dt;
		double acf = 0.;
		for (unsigned j = 0; j < t.size() - i; ++j) {
			acf += pxy[j] * pxy[i + j];
			acf += pxz[j] * pxz[i + j];
			acf += pyz[j] * pyz[i + j];
		}
		acf /= (3. * (t.size() - i));
		t_list.push_back(Dt);
		acf_list.push_back(acf);
	}

	string fn = "P_acf.txt";
	FILE* fout = fopen(fn.c_str(), "w");
	fprintf(fout, "#time(ps)\tACF(Pab)\n");
	for (unsigned i = 0; i < t_list.size(); ++i) {
		fprintf(fout, "%f\t%f\n", t_list[i], acf_list[i]);
	}
	if(string(weight_char)=="0.00") fn = "vis.txt";
	else fn = "vis-" + string(weight_char) + ".txt";
	fout = fopen(fn.c_str(), "w");
	fprintf(fout, "#time(ps)\tviscosity(mPa·s)\n");

	double convert = 6.022 * 0.001 * V / (8.314 * T);
	double vis = convert * acf_list[0] * dt / 2;
	for (unsigned i = 1; i < t_list.size(); ++i) {
		fprintf(fout, "%f\t%f\n", t_list[i] - 0.5 * dt, vis);
		if(t_list[i] <= 1){
		    vis += convert * acf_list[i] * dt;
		}
		else{
		    vis += convert * acf_list[i] * dt * pow(t_list[i], -weight);
		}
	}
	fclose(fout);
}

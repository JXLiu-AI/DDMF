#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <map>
#include <cstring>

#include "GCoptimization.h"
#include "Trimesh2/TriMesh.h"
#include "Trimesh2/Vec.h"

using namespace std;

#define PI 3.14159265

float LAMBDA = 0.25;

struct meshData{

	TriMesh *mesh;
	float* center;
	int num_labels;
	float* indicators; // num_faces * num_labels
}; 

float* findMeshCenter(TriMesh *mesh) {
	
	int numVertices = mesh->vertices.size();
	
	float* center = new float[3]();

	for(int i=0; i<numVertices; i++) {
		for (int j=0; j<3; j++) {
			center[j] += mesh->vertices[i][j];
		}
	}
	// take mean values
	for (int j=0; j<3; j++) {
		center[j] /= numVertices;
	}

	return center;
}

void writeLabel(int* res, int size, const char* path) {
  ofstream output;
  output.open (path);
  for (int i=0; i<size; i++) {
  	output << i << "," << res[i] << "\n";
  }
  output.close();
}

int annotation2label(int a, char type) {
	if (type == 'U') {
		// upper
		switch(a) {
			case 0 : return 0;
			case 11 : return 1;
			case 12 : return 2;
			case 13 : return 3;
			case 14 : return 4;
			case 15 : return 5;
			case 16 : return 6;
			case 17 : return 7;
			case 18 : return 8;
			case 21 : return 9;
			case 22 : return 10;
			case 23 : return 11;
			case 24 : return 12;
			case 25 : return 13;
			case 26 : return 14;
			case 27 : return 15;
			case 28 : return 16;
		}
	}else{
		// lower
		return 0;
	}
}

int* readLabel(const char* path, int size) {

	int* labels = new int[size]();
	std::ifstream infile(path);
	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int a, b;
		if (!(iss >> a >> b)) { break; }
		labels[a-1] = annotation2label(b, 'U');
	}
	return labels;
}

int* readNegihbors(const char* path, int pairs) {
	int* neighbors = new int[pairs*2]();
	std::ifstream infile(path);
	std::string line;
	int lc = 0;
	while (std::getline(infile, line))
	{	
		std::stringstream   linestream(line);
    	std::string         value;
    	int wc = 0;
    	while(std::getline(linestream, value, ','))
	    {
	        neighbors[lc*2+wc] = stoi(value);
	        wc ++;
	    }
	    lc ++;
	}
	return neighbors;
}

float* readLabelProb(const char* path, int num_faces, int num_labels) {

	float* probs = new float[num_faces*num_labels]();
	std::ifstream infile(path);
	std::string line;
	
	int lc = 0; 
	while (std::getline(infile, line))
	{	
		std::stringstream   linestream(line);
    	std::string         value;
    	int wc = 0; 
    	while(std::getline(linestream, value, ','))
	    {
	        probs[lc*num_labels+wc] = stod(value);
	        wc ++;
	    }
	    lc ++;
		
	}
	return probs;
}

int* pred2Labels(float* pred, int num_faces, int num_labels) {
	
	int* labels = new int[num_faces*num_labels]();
	
	for(int i=0; i<num_faces; i++) {
		int maxIdx = 0;
		for(int j=0; j<num_labels; j++) {
			if (pred[i*num_labels+maxIdx] < pred[i*num_labels+j]){
				maxIdx = j;
			}
		}
		labels[i] = maxIdx;
	}
	return labels;
}

// labeling cost, unary
float dataCost(int p, int l, void *data) {
	meshData *mdata = (meshData *) data;
	int nl = mdata->num_labels;
	return -log(mdata->indicators[p*nl+l] + 1e-6);
}

float* computePlane(TriMesh* mesh, int p) {

	float* f = new float[3]();

	point  pt1 = mesh->vertices[mesh->faces[p][0]] - mesh->vertices[mesh->faces[p][1]];
	point  pt2 = mesh->vertices[mesh->faces[p][2]] - mesh->vertices[mesh->faces[p][1]];
	
	f[0] = pt1[1]*pt2[2] - pt1[2]*pt2[1];
	f[1] = pt1[2]*pt2[0] - pt1[0]*pt2[2];
	f[2] = pt1[0]*pt2[1] - pt1[1]*pt2[0];

	return f;
}

// compute distance
float distance(TriMesh* mesh, int p1, int p2) {

	point dist_center = mesh->faces[p1].faceCenter - mesh->faces[p2].faceCenter;

	float dist = 0.0;
	for (int i=0; i<3; i++) {
		dist += pow(dist_center[i], 2);
	}
	return sqrt(dist);
}

// compute dihedral angle
float dihedralAngle(TriMesh* mesh, int p1, int p2) {

	float* f1 = computePlane(mesh, p1);
	float* f2 = computePlane(mesh, p2);

	float dot = f1[0]*f2[0] + f1[1]*f2[1] + f1[2]*f2[2];
	float norms = pow((f1[0]*f1[0]+f1[1]*f1[1]+f1[2]*f1[2]), 0.5) * pow((f2[0]*f2[0]+f2[1]*f2[1]+f2[2]*f2[2]), 0.5);

	float cosine = abs(dot) / (norms + 1e-20);

    if (cosine > 1) 
    	cosine = 1;
    else if (cosine < -1) 
    	cosine = -1;
	
	return acos(cosine);
}

// smooth cost, pair
float smoothCost(int p1, int p2, int l1, int l2, void *data) {

	if ( l1 == l2 ) return 0;

	meshData *mdata = (meshData *) data;
	TriMesh *mesh = mdata->mesh;
	// compute distance
	float phi = distance(mesh, p1, p2);
	// compute angel
	float theta = dihedralAngle(mesh, p1, p2);
	return -LAMBDA*log((theta+1e-20)/PI)*phi;
}


int* optimize(int nf, int nl, int* labels, float* indicators, int* neighbors, int pairs, TriMesh* mesh) {

	// create general graph
	int *result = new int[nf]; 
	int nv = mesh->vertices.size();

	try {

		meshData mdata;
		mdata.mesh = mesh;
		mdata.center = findMeshCenter(mesh);
		mdata.num_labels = nl;
		mdata.indicators = indicators;
			
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(nf, nl);

		cout << "set initial labels" << endl;
		//set up labels
		for (int i=0; i<nf; i++) {
			gc->setLabel(i, labels[i]);
		}
		cout << "set neighbor system" << endl;
		// set up neighbor system
		for (int i=0; i<pairs; i++) {
			gc->setNeighbors(neighbors[i*2], neighbors[i*2+1]);
		}
		cout << "set data cost" << endl;
		// set data cost
		gc->setDataCost(&dataCost, &mdata);
		cout << "set smooth cost" << endl;
		// set smooth cost
		gc->setSmoothCost(&smoothCost, &mdata);
		cout << "swapping ..." << endl;
		printf("Before optimization energy is %f\n", gc->compute_energy());
		gc->swap();
		printf("After optimization energy is %f\n", gc->compute_energy());
		cout << "printing ..." << endl;
		for ( int  i = 0; i < nf; i++ )
			result[i] = gc->whatLabel(i);

	}catch (GCException e){
		e.Report();
	}

	return result;
}

int main(int argc, char **argv) {

	LAMBDA = stof(argv[1]);
	printf("lambda = %f\n", LAMBDA);
	// number of labels
	const int nl = stoi(argv[2]);
	// mesh file path
	const char *meshFileName = argv[3];
	// pred prob file path
	const char *predFileName = argv[4];
	// neighbor pair file path
	const char *neighborFileName = argv[5];
	// output label file path
	const char *output = argv[6];

	const int numNeighborPairs = stoi(argv[7]);

	// cout << "Reading mesh file:" << meshFileName << endl;
	TriMesh* mesh = TriMesh::read(meshFileName);
	mesh->need_normals();
	mesh->need_adjacentfaces();
	mesh->need_across_edge();

	int nf = mesh->faces.size();
	// int nl = 17;

	cout << "Reading labels file: " << predFileName << endl;
	float *preds = readLabelProb(predFileName, nf, nl);
	int *labels = pred2Labels(preds, nf, nl);

	cout << "Reading neighbors file: " << neighborFileName << endl;
	int* neighbors = readNegihbors(neighborFileName, numNeighborPairs);

	cout << "Optimizing ..." << endl;
	int* results = optimize(nf, nl, labels, preds, neighbors, numNeighborPairs, mesh);

	cout << "Output labels ..." << endl;
	writeLabel(results, nf, output);
}

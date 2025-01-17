#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAXN 1000000

//todo: read clusters from file
//todo: the choise for init clusters
//todo: the ending criteria



int N; // number of points
int K; // number of clusters
int T; // number of iterations
char INPUT_FILE[256]; // input file name

float *POINTS; // POINTS[i*2+0]:x POINTS[i*2+1]:y
int *CLASSES; // class for each point
int *NUM_CLASSES; // number of points in each class
float *CLUSTERS; // position for each cluster


void write_results(int n, int k){
    FILE *outputFile;
    int i;
    
    outputFile = fopen("Classes.txt", "w");
    for(i=0;i<n;i++){
        fprintf(outputFile, "%d\n", CLASSES[i]);
    }
    fclose(outputFile);
    
    outputFile = fopen("Clusters.txt", "w");
    for(i=0;i<k;i++){
        fprintf(outputFile, "%f,%f\n", CLUSTERS[i*2], CLUSTERS[i*2+1]);
    }
    fclose(outputFile);    
}

void update_classes(int n, int k){ //based on CLUSTERS
    int i,j,minK;
    float minDis, dis, disX, disY;
    for(i=0;i<n;i++){
        disX = POINTS[i*2]-CLUSTERS[0];
        disY = POINTS[i*2+1]-CLUSTERS[1];
        minK = 0;
        minDis = disX*disX + disY*disY;
        for(j=1;j<k;j++){
            disX = POINTS[i*2]-CLUSTERS[j*2];
            disY = POINTS[i*2+1]-CLUSTERS[j*2+1];
            dis = disX*disX + disY*disY;
            if(dis<minDis){
                minK = j;
                minDis = dis;
            }
        }
        CLASSES[i] = minK;
    }
}
    
void update_clusters(int n, int k){ // based on CLASSES
    int i;
	int _class;
    for(i=0;i<k;i++){
        CLUSTERS[i*2]=0;
        CLUSTERS[i*2+1]=0;
        NUM_CLASSES[i]=0;
    }
    for(i=0;i<n;i++){
        _class = CLASSES[i];
        NUM_CLASSES[_class]++;
        CLUSTERS[_class*2] += POINTS[i*2];
        CLUSTERS[_class*2+1] += POINTS[i*2+1];
    }
    for(i=0;i<k;i++){
        //if(NUM_CLASSES[i]!=0){
            CLUSTERS[i*2] /= NUM_CLASSES[i]; // produce nan when divided by 0
            CLUSTERS[i*2+1] /= NUM_CLASSES[i];
        //}
    }    
}
    
void clean_clusters(int *K){ // remove empty clusters, CLASSES are invalid after this process
    int i;
    for(i=0;i<*K;i++){
        if(NUM_CLASSES[i]==0){
            CLUSTERS[i*2] = CLUSTERS[*K * 2];
            CLUSTERS[i*2+1] = CLUSTERS[*K * 2 + 1];
            *K--;
            i--; // the new cluster is not tested
        }
    }
}

void init(int n, int k, char *input){
    FILE *inputFile;
    int i;
    float x,y;
    
    // read points
    POINTS = (float*)malloc(n * 2 * sizeof(float));
    inputFile = fopen(input, "r");
    for(i=0;i<n;i++){
        if(fscanf(inputFile, "%f,%f\n", &x, &y)==2){
            POINTS[i*2] = x;
            POINTS[i*2+1] = y;
        }
    }
    fclose(inputFile);
    
    // classes init
    CLASSES = (int*)malloc(n * sizeof(int));
    
    // clusters init
    NUM_CLASSES = (int*)malloc(k * sizeof(int));
    CLUSTERS = (float*)malloc(k * 2 * sizeof(float));
    for(i=0;i<k;i++){
        CLUSTERS[i*2]=POINTS[i*2];
        CLUSTERS[i*2+1]=POINTS[i*2+1];
    }    
}

int data_count(char *fileName){
    FILE *inputFile;
    float x, y;
    int count=0;
    inputFile = fopen(fileName, "r");
    while(fscanf(inputFile, "%f,%f\n", &x, &y)==2){
        count++;    
        //printf("%f,%f\n",tmp1,tmp2);
    }
    fclose(inputFile);
    return count;
}


int cmd_parser(int argc, char **argv, int *n, int *k, int *t, char *input){
    int invalid;
    int valid;
    char ch;
    char usage[] = "Usage: %s -n N -k K -t T -i Input.txt\n"
                   "    N: Number_of_Points, default: the number of lines in Input_File\n"
                   "    K: default: 2\n"
                   "    T: max iterations for the kmeans algorithm\n"
                   "    Input: should be n lines, two floats in each line and split by ','\n"
		           "    Results will be in Classes.txt and Clusters.txt\n";
    invalid = 0;
    valid = 0;
    if(argc==1){
        invalid = 1;
    }
    
    //default values
    *n = -1;
    *k = 2;
    *t = 1;
    
    while((ch = getopt(argc, argv, "n:k:t:i:h")) != -1) {
        switch(ch) {
            case 'n':
                sscanf(optarg, "%d", n);
                break;
            case 'k':
                sscanf(optarg, "%d", k);
                break;
            case 't':
                sscanf(optarg, "%d", t);
                break;
            case 'i':
                strncpy(input, optarg, 256);
                valid = 1;
                break;
            case 'h':  //print help
                invalid = 1;
                break;
            case '?':
                invalid = 1;
            default:
                ;
        }
    }
	
    
    if(valid && *n==-1){
        *n = data_count(input);
    }
    
    
    invalid = invalid || !valid;
    if(invalid){
        printf(usage, argv[0]);
    }
    
    if(*n>MAXN){
        invalid = 1;
        printf("N is too large\n");
    }
    
    
    //printf("option N: %d\n", *n);
    //printf("option K: %d\n", *k);
    //printf("option T: %d\n", *t);
    //printf("option Input: %s\n", input);
    //printf("invalid %d\n", invalid);
	
    return invalid;    
}


int main(int argc, char **argv) {
    int t;
    if(cmd_parser(argc, argv, &N, &K, &T, INPUT_FILE)){ // not enough parameters
        return 1;
    }
    init(N, K, INPUT_FILE);
	update_classes(N, K);
    for(t=0;t<T;t++){
		if(t!=0){
			clean_clusters(&K);
		}
        update_classes(N, K);
        update_clusters(N, K);
    }
	
    write_results(N, K);
    return 0;
}

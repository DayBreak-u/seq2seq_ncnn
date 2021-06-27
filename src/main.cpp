#include "seq2seq.h"
#include<stdio.h>

int main(int argc, char **argv) {

    Sqeseq seq2seqnet;
    seq2seqnet.initModel();
    seq2seqnet.setNumThread(1);

    char str[1000];
    while (true){
        printf("question:");
        if(scanf("%s",&str) == EOF) break;
        std::string intput = str;
//        printf("111111%s\n",intput.c_str());
        std::vector<std::string> res = seq2seqnet.forward(intput);
         printf("answer:");
        for (int i = 0; i< res.size();i++)
        {
            printf("%s",res[i].c_str());
        }
        printf("\n");

    }

    return 0;
}
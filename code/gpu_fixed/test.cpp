 
// Includes
#include  <stdio.h>
#include  <stdlib.h>
#include  <iostream>
#include  <cstring>
//#include  <math.h>
//#include  <time.h>
#include  <string.h>
//#include  <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <builtin_types.h>
//#include <immintrin.h>
//#include <emmintrin.h>
//#include <xmmintrin.h>

using namespace std;

#include <omp.h>

//#include "CFloodingGpuDecoder.h"

#include "./decoder_ms/CGPU_Decoder_MS_SIMD.h"
#include "./decoder_oms/CGPU_Decoder_OMS_SIMD.h"
#include "./decoder_nms/CGPU_Decoder_NMS_SIMD.h"
#include "./decoder_2nms/CGPU_Decoder_2NMS_SIMD.h"
#include "./decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h"


//#define pi  3.1415926536

#include "./timer/CTimer.h"
#include "./trame/CTrame.h"
#include "./awgn_channel/CChanel_AWGN_SIMD.h"
#include "./ber_analyzer/CErrorAnalyzer.h"
#include "./terminal/CTerminal.h"

#include "./matrix/constantes_gpu.h"

//#define SINGLE_THREAD 1

int    QUICK_STOP           =  false;
bool   BER_SIMULATION_LIMIT =  false;
double BIT_ERROR_LIMIT      =  1e-7;

//int technique          = 0;
//int sChannel           = 1; // CHANNEL ON GPU

////////////////////////////////////////////////////////////////////////////////////

void show_info(){
	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
//  	printf("(II) Identifiant du GPU (CUDA)    : %s\n", devProp.name);
  	printf("(II) Nombre de Multi-Processor    : %d\n", devProp.multiProcessorCount);
  	printf("(II) + totalGlobalMem             : %ld Mo\n", (devProp.totalGlobalMem/1024/1024));
  	printf("(II) + sharedMemPerBlock          : %ld Ko\n", (devProp.sharedMemPerBlock/1024));
#ifdef CUDA_6
  	printf("(II) + sharedMemPerMultiprocessor : %ld Ko\n", (devProp.sharedMemPerMultiprocessor/1024));
  	printf("(II) + regsPerMultiprocessor      : %ld\n", devProp.regsPerMultiprocessor);
#endif
  	printf("(II) + regsPerBlock               : %d\n", (int)devProp.regsPerBlock);
  	printf("(II) + warpSize                   : %d\n", (int)devProp.warpSize);
  	printf("(II) + memoryBusWidth             : %d\n", (int)devProp.memoryBusWidth);
  	printf("(II) + memoryClockRate            : %d\n", (int)devProp.memoryClockRate);

  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_MS_SIMD);
  	  	printf("(II) CGPU_Decoder_MS_SIMD   (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_OMS_SIMD);
  	  	printf("(II) CGPU_Decoder_OMS_SIMD   (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_NMS_SIMD);
  	  	printf("(II) CGPU_Decoder_NMS_SIMD  (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_2NMS_SIMD);
  	  	printf("(II) CGPU_Decoder_2NMS_SIMD (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
	fflush(stdout);
}

////////////////////////////////////////////////////////////////////////////////////

#define MAX_THREADS 4

int main(int argc, char* argv[])
{
    int p;
    srand( 0 );
	printf("(II) LDPC DECODER - Flooding scheduled decoder\n");
	printf("(II) MANIPULATION DE DONNEES (IEEE-754 - %ld bits)\n", (long int)8*sizeof(int));
	printf("(II) GENEREE : %s - %s\n", __DATE__, __TIME__);

	double Eb_N0;
	double MinSignalSurBruit  = 0.50;
	double MaxSignalSurBruit  = 1;
	double PasSignalSurBruit  = 0.10;
    int    NOMBRE_ITERATIONS  = 20;
	int    STOP_TIMER_SECOND  = -1;
	bool   QPSK_CHANNEL       = false;
    bool   Es_N0              = false; // FALSE => MODE Eb_N0
    int    NB_THREAD_ON_GPU   = 1024;
	int    FRAME_ERROR_LIMIT  =  200;

	char  defDecoder[] = "fMS";
    const char* type = defDecoder;
	
	    //
    // ON CONFIGURE LE NOMBRE DE THREAD A UTILISER PAR DEFAUT
    //
    int NUM_ACTIVE_THREADS = 1;
    omp_set_num_threads(NUM_ACTIVE_THREADS);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();

	//
	// ON VA PARSER LES ARGUMENTS DE LIGNE DE COMMANDE
	//
	for (p=1; p<argc; p++) {
		if( strcmp(argv[p], "-min") == 0 ){
			MinSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-max") == 0 ){
			MaxSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-pas") == 0 ){
			PasSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-timer") == 0 ){
			STOP_TIMER_SECOND = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-iter") == 0 ){
			NOMBRE_ITERATIONS = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-fer") == 0 ){
			FRAME_ERROR_LIMIT = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-qef") == 0 ){
			BER_SIMULATION_LIMIT =  true;
			BIT_ERROR_LIMIT      = ( atof( argv[p+1] ) );
			p += 1;

		}else if( strcmp(argv[p], "-bpsk") == 0 ){
			QPSK_CHANNEL = false;

		}else if( strcmp(argv[p], "-qpsk") == 0 ){
			QPSK_CHANNEL = true;

		}else if( strcmp(argv[p], "-Eb/N0") == 0 ){
			Es_N0 = false;

		}else if( strcmp(argv[p], "-Es/N0") == 0 ){
			Es_N0 = true;

		}else if( strcmp(argv[p], "-n") == 0 ){
			NB_THREAD_ON_GPU = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-fMS") == 0 ){
			type      = "fMS";

		}else if( strcmp(argv[p], "-xMS") == 0 ){
			type      = "xMS";

		}else if( strcmp(argv[p], "-MS") == 0 ){
			type      = "MS";

		}else if( strcmp(argv[p], "-OMS") == 0 ){
			type      = "OMS";

		}else if( strcmp(argv[p], "-NMS") == 0 ){
			type      = "NMS";

		}else if( strcmp(argv[p], "-2NMS") == 0 ){
			type      = "2NMS";
			p+=1;
		}else if (strcmp(argv[p], "-thread") == 0){
            int nThreads = atoi(argv[p + 1]);
            if (nThreads > 4) {
                printf("(WW) Number of thread can be higher than 4 => Using 4 threads.");
                NUM_ACTIVE_THREADS = 4;
            } else if (nThreads < 1) {
                printf("(WW) Number of thread can be lower than 1 => Using 1 thread.");
                NUM_ACTIVE_THREADS = 1;
            } else {
                NUM_ACTIVE_THREADS = nThreads;
            }
            omp_set_num_threads(NUM_ACTIVE_THREADS);
            p += 1;
		}else if( strcmp(argv[p], "-info") == 0 ){
			show_info();
			exit( 0 );

		}else{
			printf("(EE) Unknown argument (%d) => [%s]\n", p, argv[p]);
			exit(0);
		}
	}

	double rendement = (double)(NmoinsK)/(double)(_N);
	printf("(II) Code LDPC (N, K)     : (%d,%d)\n", _N, _K);
	printf("(II) Rendement du code    : %.3f\n", rendement);
	printf("(II) # ITERATIONs du CODE : %d\n", NOMBRE_ITERATIONS);
    printf("(II) FER LIMIT FOR SIMU   : %d\n", FRAME_ERROR_LIMIT);
	printf("(II) SIMULATION  RANGE    : [%.2f, %.2f], STEP = %.2f\n", MinSignalSurBruit,  MaxSignalSurBruit, PasSignalSurBruit);
	printf("(II) MODE EVALUATION      : %s\n", ((Es_N0)?"Es/N0":"Eb/N0") );
	printf("(II) MIN-SUM ALGORITHM    : %s\n", type );
	printf("(II) FAST STOP MODE       : %d\n", QUICK_STOP);
        
        CTimer simu_timer(true);
        
        
     //
    // ALLOCATION DYNAMIQUE DES DONNESS NECESSAIRES A LA SIMULATION DU SYSTEME
    //
    CTrame* simu_data[MAX_THREADS];
    for(int i=0; i<4; i++)
    {
        simu_data[i] = new CTrame(_N, _K, NB_THREAD_ON_GPU);
    }

    CGPUDecoder* decoder[MAX_THREADS];
	if( strcmp(type, "fMS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_MS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else if( strcmp(type, "MS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_MS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else if( strcmp(type, "OMS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_OMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else if( strcmp(type, "NMS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_NMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else if( strcmp(type, "2NMS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_2NMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else if( strcmp(type, "xMS") == 0 )
	{
		for(int i=0;i<4;i++)
			decoder[i] = new CGPU_Decoder_MS_SIMD_v2( NB_THREAD_ON_GPU, _N, _K, _M );
	}
	else
	{
		printf("(EE) Requested decoder does not exist !\n");
		exit( 0 );
	}

    for(int i=0; i<4; i++)
    {
        decoder[i]->initialize();
    }
        

    CChanel_AWGN_SIMD* noise[MAX_THREADS];
    for(int i=0; i<4; i++)
    {
        noise[i] = new CChanel_AWGN_SIMD(simu_data[i], 4, QPSK_CHANNEL, Es_N0);
    }
        
        
    Eb_N0 = MinSignalSurBruit;
    long int temps = 0, fdecoding = 0;
        
    CErrorAnalyzer* errCounter[MAX_THREADS];
    
	long int etime[MAX_THREADS] = {0, 0, 0, 0};
	if(STOP_TIMER_SECOND == -1)
    while (Eb_N0 <= MaxSignalSurBruit)
    {

		for(int i=0; i<4; i++)
		{
            noise[i]->configure(Eb_N0);
        }
        //
        // ON CREE UN OBJET POUR LA MESURE DU TEMPS DE SIMULATION (REMISE A ZERO POUR CHAQUE Eb/N0)
        //
        CTimer temps_ecoule(true);
		CTimer term_refresh(true);

        CErrorAnalyzer errCounters(simu_data[0], FRAME_ERROR_LIMIT, false, false);
        for(int i=0; i<4; i++)
        {
            errCounter[i] = new CErrorAnalyzer(simu_data[i], FRAME_ERROR_LIMIT, true, true);
        }
        
        //
        //
        // ON CREE L'OBJET EN CHARGE DES INFORMATIONS DANS LE TERMINAL UTILISATEUR
        //
        CTerminal terminal(&errCounters, &temps_ecoule, Eb_N0);
            
        //
        
        // ON GENERE LA PREMIERE TRAME BRUITEE

        for(int i=0; i<4; i++)
        {
            noise[i]->generate();
        }

        for(int i=0; i<4; i++)
        {
            errCounter[i]->store_enc_bits();
        }
        
//         CTimer timer[MAX_THREADS];
        

//         CTimer biais_measure(false);
//         biais_measure.start();
//         biais_measure.stop();
//         unsigned int biais = biais_measure.get_time_us();

        //
        
        #pragma omp parallel sections
        {         
            #pragma omp section
            {
                CTimer essai(true);
				decoder[0]->decode_stream( simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NOMBRE_ITERATIONS );
                etime[0] += essai.get_time_us();
                noise[0]->generate();  // ON GENERE LE BRUIT DU CANAL
                errCounter[0]->generate();
				#pragma omp atomic
                fdecoding += 1;
            }
            
            #pragma omp section
            {
                CTimer essai(true);
                decoder[1]->decode_stream( simu_data[1]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NOMBRE_ITERATIONS );
                etime[1] += essai.get_time_us();
                noise[1]->generate();  // ON GENERE LE BRUIT DU CANAL
                errCounter[1]->generate();
				#pragma omp atomic
                fdecoding += 1;
            }
            
            #pragma omp section
            {
                CTimer essai(true);
				decoder[2]->decode_stream( simu_data[2]->get_t_noise_data(), simu_data[2]->get_t_decode_data(), NOMBRE_ITERATIONS );
                etime[2] += essai.get_time_us();
                noise[2]->generate();  // ON GENERE LE BRUIT DU CANAL
                errCounter[2]->generate();
				#pragma omp atomic
                fdecoding += 1;
            }
            
            #pragma omp section
            {
                CTimer essai(true);
                decoder[3]->decode_stream( simu_data[3]->get_t_noise_data(), simu_data[3]->get_t_decode_data(), NOMBRE_ITERATIONS );
                etime[3] += essai.get_time_us();
                noise[3]->generate();  // ON GENERE LE BRUIT DU CANAL
                errCounter[3]->generate();
				#pragma omp atomic
                fdecoding += 1;
            }
            
        }
        
        //
        // ON COMPTE LE NOMBRE D'ERREURS DANS LA TRAME DECODE
        //
        errCounters.reset_internals();
        errCounters.accumulate( errCounter[0]);

        //
        // ON compare le Frame Error avec la limite imposee par l'utilisateur. Si on depasse
        // alors on affiche les resultats sur Eb/N0 courant.
        //
//         if ( errCounters.fe_limit_achieved() == true )
//         {
// 			std::cerr <<  "got out of loop !!" << __LINE__ << std::endl;
//             break;
//         }
        
      	if( term_refresh.get_time_sec() >= 1 )
		{
			term_refresh.reset();
	        terminal.temp_report();
		}
        
        if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) )
        {
            printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) TIME CONTRAINT.\n");
            printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps/1000);
                    temps /= fdecoding;
            printf("(II) + TIME / RUN = %dms\n", temps/1000);
            int   workL = 4 * NB_THREAD_ON_GPU;
            int   kbits = workL * _N / temps ;
            float mbits = ((float)kbits) / 1000.0;
            printf("(II) + DECODER LATENCY (ms)     = %d\n", temps);
            printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
            printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", Eb_N0, NB_THREAD_ON_GPU, workL, NOMBRE_ITERATIONS, mbits);
                cout << endl << "Temps = " << temps << "ms : " << kbits;
                cout << "kb/s : " << ((float)temps/NB_THREAD_ON_GPU) << "ms/frame" << endl << endl;
			std::cerr <<  "got out of loop !!" << __LINE__ << std::endl;
            break;
        }
        
        terminal.final_report();

        if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) )
		{
			std::cerr <<  "got out of loop !!" << __LINE__ << std::endl;
        	break;
        }

		Eb_N0 = Eb_N0 + PasSignalSurBruit;

        if( BER_SIMULATION_LIMIT == true )
		{
        	if( errCounters.ber_value() < BIT_ERROR_LIMIT ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) QUASI-ERROR FREE CONTRAINT.\n");
				std::cerr <<  "got out of loop !!" << __LINE__ << std::endl;
        		break;
        	}
        }
	}
	
	    if(STOP_TIMER_SECOND==-1)
        {
            printf("FINAL REPORT.\n");
// 			int tempDum = 0;
// 			for(int i=0;i<4.i++)
			temps = etime[0];
            printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps/1000);
                    temps /= fdecoding;
            printf("(II) + TIME / RUN = %dms\n", temps/1000);
            int   workL = NUM_ACTIVE_THREADS * NB_THREAD_ON_GPU;
			int flt = sizeof(float);
			temps /= 1000;
            float   kbits = ((float)(workL * _N / temps) );
            float mbits = ((float)kbits/1000.0);
            printf("(II) + DECODER LATENCY (ms)     = %d\n", temps/1000);
            printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
            printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", Eb_N0, NB_THREAD_ON_GPU, workL, NOMBRE_ITERATIONS, mbits);
                cout << endl << "Temps = " << temps/1000 << "ms : " << mbits*1000;
                cout << "kb/s : " << ((float)(temps/1000)/NB_THREAD_ON_GPU) << "ms/frame" << endl << endl;
        }

        if(0)
        {
            printf("FINAL REPORT.\n");
// 			int tempDum = 0;
// 			for(int i=0;i<4.i++)
			temps = etime[0];
            printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps/1000);
                    temps /= fdecoding;
            printf("(II) + TIME / RUN = %dms\n", temps/1000);
            int   workL = 4 * NB_THREAD_ON_GPU;
            int   kbits =  (workL * _N * 1000 * NUM_ACTIVE_THREADS)/ (temps) ;
			printf("TIME = %ld\n", temps);
			printf("KBPS = %d\n", kbits);
            float mbits = ((float)kbits) / 1000.0;
            printf("(II) + DECODER LATENCY (ms)     = %d\n", temps/1000);
            printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
            printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", Eb_N0, NB_THREAD_ON_GPU, workL, NOMBRE_ITERATIONS, mbits);
                cout << endl << "Temps = " << temps << "ms : " << kbits;
                cout << "kb/s : " << ((float)temps/NB_THREAD_ON_GPU) << "ms/frame" << endl << endl;
        }
      
      
      ////////////////////////////////////////////////////////////////////////////////
    //
    //
    // SECOND EVALUATION OF THE THROUGHPUT WITHOUT ENCODED FRAME REGENERATION
    //
    //
    if( STOP_TIMER_SECOND != -1 )
    {
        int exec = 0;
        const int t_eval = STOP_TIMER_SECOND;


        //
        // ONE THREAD MODE
        //
        if (NUM_ACTIVE_THREADS == 1) 
		{
            CTimer t_Timer1(true);
            while (t_Timer1.get_time_sec() < t_eval) 
			{
                for (int qq = 0; qq < 10; qq++) 
				{
                    // to limit timer runtime impact on performances (for very small LDPC codes)
                    // Indeed, depending on OS and CTimer implementations, time read can be long...
                    decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    exec += 1;
                }
            }
//             t_Timer1.stop();
            float debit = _N * ((exec ) / ((float) t_Timer1.get_time_sec()));
            debit /= 1000000.0f;
            printf("(PERF1) LDPC decoder air throughput = %1.6f Mbps\n", debit);
        }

        //
        // TWO THREAD MODE
        //
        if (NUM_ACTIVE_THREADS == 2) {
            exec = 0;
            omp_set_num_threads(2);
            CTimer t_Timer2(true);

            while (t_Timer2.get_time_sec() < t_eval) 
			{
                const int looper = 256;
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[1]->decode(simu_data[1]->get_t_noise_data(), simu_data[2]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                }
                exec += 2 * looper;
            }
            t_Timer2.stop();

            // for each decoder run, we decoded nb_frames codewords (depending on the SIMD width)
            float debit = _N * ((exec) / ((float) t_Timer2.get_time_sec()));
            debit /= 1000000.0f;
            printf("(PERF2) LDPC decoder air throughput = %1.3f Mbps\n", debit);
        }


        //
        // FOUR THREAD MODE
        //
        if (NUM_ACTIVE_THREADS == 4) 
		{
            exec = 0;
            omp_set_num_threads(4);
            CTimer t_Timer3(true);

            while (t_Timer3.get_time_sec() < t_eval) 
			{
                const int looper = 256;
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[1]->decode(simu_data[1]->get_t_noise_data(), simu_data[2]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                    #pragma omp section
                        {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[2]->decode(simu_data[2]->get_t_noise_data(), simu_data[3]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[3]->decode(simu_data[3]->get_t_noise_data(), simu_data[4]->get_t_decode_data(), NOMBRE_ITERATIONS);
                    }
                }
                exec += 4 * looper;
            }
            t_Timer3.stop();

            float debit = _N * ((exec) / ((float) t_Timer3.get_time_sec()));
            debit /= 1000000.0f;
            printf("(PERF4) LDPC decoder air throughput = %1.3f Mbps\n", debit);
        }
        exit(0);
    }
    
    
 return 0;
}

#include "handler.h"

WorkItem::WorkItem(CTrame * data, CChanel_AWGN_SIMD* noise) : 
simData(data), noise(noise)
{
	
}
WorkItem::~WorkItem() 
{
	if(simData) {delete simData;}  
	if(noise) {delete noise;}
	simData = NULL;
	noise = NULL;
}

Worker::Worker(wqueue<WorkItem*>& queue, const char* DecoderType, int numThreadOnGpu, int FERL, int numberIter) : 
m_queue(queue), numberThreadOnGpu(numThreadOnGpu), frameErrorLimit(FERL), numberIter(numberIter)
{
	if( strcmp(DecoderType, "fMS") == 0 )
	{
		decoder = new CGPU_Decoder_MS_SIMD( numThreadOnGpu, _N, _K, _M );
	}
	else if( strcmp(DecoderType, "MS") == 0 )
	{
		decoder = new CGPU_Decoder_MS_SIMD( numThreadOnGpu, _N, _K, _M );
	}
	else if( strcmp(DecoderType, "OMS") == 0 )
	{
		decoder = new CGPU_Decoder_OMS_SIMD( numThreadOnGpu, _N, _K, _M );
	}
	else if( strcmp(DecoderType, "NMS") == 0 )
	{
		decoder = new CGPU_Decoder_NMS_SIMD( numThreadOnGpu, _N, _K, _M );
	}
	else if( strcmp(DecoderType, "2NMS") == 0 )
	{
		decoder = new CGPU_Decoder_2NMS_SIMD( numThreadOnGpu, _N, _K, _M );
	}
	else if( strcmp(DecoderType, "xMS") == 0 )
	{
		decoder = new CGPU_Decoder_MS_SIMD_v2( numThreadOnGpu, _N, _K, _M );
	}
	else
	{
		printf("(EE) Requested decoder does not exist !\n");
// 			exit( 0 );
	}
	decoder->initialize();
}

void* Worker::run() 
{
	for (int i = 0;; i++) 
	{
		printf("thread %lu, loop %d - waiting for item...\n",  (long unsigned int)self(), i);
		WorkItem* item = m_queue.remove();
		printf("thread %lu, loop %d - got one item\n",  (long unsigned int)self(), i);
		{
			int temps = 0, fdecoding = 0;
					//
			// ON CREE UN OBJET POUR LA MESURE DU TEMPS DE SIMULATION (REMISE A ZERO POUR CHAQUE Eb/N0)
			//
			CTimer temps_ecoule(true);
			CTimer term_refresh(true);
			CErrorAnalyzer errCounters(item->getData(), frameErrorLimit, false, false);
			CErrorAnalyzer errCounter (item->getData(), frameErrorLimit, true,  true);

			//
			// ON CREE L'OBJET EN CHARGE DES INFORMATIONS DANS LE TERMINAL UTILISATEUR
			//
			CTerminal terminal(&errCounters, &temps_ecoule, item->getNoise()->getEb_N0());

			//
			// ON GENERE LA PREMIERE TRAME BRUITEE
			//
			item->getNoise()->generate();
			errCounter.store_enc_bits();

			while( 1 )
			{
				//
				//	ON LANCE LE TRAITEMENT SUR PLUSIEURS THREAD...
				//
// 					CTimer essai(true);
// 					decoder->decode( item->getData()->get_t_noise_data(), item->getData()->get_t_decode_data(), numberIter );
// 					temps += essai.get_time_ms();
				fdecoding += 1;
				#pragma omp sections
				{
					#pragma omp section
					{
// 							item->getNoise()->generate();
					}
					#pragma omp section
					{
// 							errCounter.generate();
					}
				}

				//
				// ON COMPTE LE NOMBRE D'ERREURS DANS LA TRAME DECODE
				//
// 					errCounters.reset_internals();
// 					errCounters.accumulate( &errCounter );

				//
				// ON compares the frame error with the limits imposed by the user. 
				// If it exceeds then displays the results on Eb / N0 current.
				//
// 					if ( errCounters.fe_limit_achieved() == true ){
// 					break;
// 					}

				//
				// AFFICHAGE A L'ECRAN DE L'EVOLUTION DE LA SIMULATION SI NECESSAIRE
				//
// 					if( term_refresh.get_time_sec() >= 1 ){
// 						term_refresh.reset();
// 						terminal.temp_report();
// 					}

// 					if( (simTotalTimer->get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) )
// 					{
// 						printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) TIME CONTRAINT.\n");
// 						printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps);
// 						temps /= fdecoding;
// 						printf("(II) + TIME / RUN = %dms\n", temps);
// 						int   workL = 4 * numberThreadOnGpu;
// 						int   kbits = workL * _N / temps ;
// 						float mbits = ((float)kbits) / 1000.0;
// 						printf("(II) + DECODER LATENCY (ms)     = %d\n", temps);
// 						printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
// 						printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", item->getNoise()->getEb_N0(), numberThreadOnGpu, workL, numberIter, mbits);
// 						cout << endl << "Temps = " << temps << "ms : " << kbits;
// 						cout << "kb/s : " << ((float)temps/numberThreadOnGpu) << "ms/frame" << endl << endl;
// 						break;
// 					}
			}

// 				terminal.final_report();
			
			
		}
		delete item;
	}
	return NULL;
}
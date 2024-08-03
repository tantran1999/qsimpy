// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg meas[12];
u3(1.4368347742816616,-3.1167849646094092,-pi) q[0];
u3(0.13038834331032637,-3.064986403423018,0) q[1];
cx q[0],q[1];
u3(2.3018560275705338,1.9642554329689128,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(1.5783117544539513,0.70702212977712,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
u3(3.1322119352256292,-1.7482629013133648,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(1.4124389803026798,1.8339114230470699,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(1.2444656817555668,-0.5166404252966217,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
u3(1.5046299106322625,1.3482194095209188,-pi) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
u3(1.0625547235745711,-2.8742785055981948,0) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
u3(0.5550554224571163,0.8932807542109362,0) q[9];
cx q[0],q[9];
cx q[1],q[9];
cx q[2],q[9];
cx q[3],q[9];
cx q[4],q[9];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
u3(1.9769425662797728,-0.7958234754631421,-pi) q[10];
cx q[0],q[10];
cx q[1],q[10];
cx q[2],q[10];
cx q[3],q[10];
cx q[4],q[10];
cx q[5],q[10];
cx q[6],q[10];
cx q[7],q[10];
cx q[8],q[10];
cx q[9],q[10];
u3(0.2928382424047807,1.0941137716709264,-pi) q[11];
cx q[0],q[11];
u3(2.7761197097590844,0.716184867709297,0) q[0];
cx q[1],q[11];
u3(2.72699034602209,-1.0764269733891005,0) q[1];
cx q[0],q[1];
cx q[2],q[11];
u3(2.4016409048004452,-2.846934388642458,-pi) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[11];
u3(3.059042641009883,0.7934855547557511,-pi) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[11];
u3(2.1966192898367836,0.2989926356969477,-pi) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[11];
u3(2.5067461861055573,2.0061393599676878,-pi) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[11];
u3(1.2238187478398983,-1.891568395380352,-pi) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[11];
u3(3.005579583727834,2.2421565772650203,-pi) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[11];
u3(0.5739760098973872,-0.9320939562791777,-pi) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[11];
cx q[10],q[11];
u3(0.5683728542359917,1.8595822481541884,0) q[10];
u3(1.8893541777246623,-0.7292486063206916,0) q[11];
u3(2.0058195038543025,-1.5415940196621944,0) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
u3(2.0452499401435484,-2.5297885440896417,0) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
u3(1.0368254640000032,-0.6105260558088226,0) q[1];
cx q[0],q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
u3(2.4663339782035085,-2.9244586574319644,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
u3(0.5872288522304123,-2.574840774992211,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
u3(1.124026302216569,-2.894778030919191,-pi) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
u3(0.9497161489686776,2.244239177845083,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
u3(2.4134622602982154,0.5002237983271183,0) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
u3(0.35022027389382004,-1.2223313827259217,-pi) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
u3(0.07775143427988175,-1.0636219317431195,-pi) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
u3(1.0925023928214677,-2.890521540662405,-pi) q[10];
u3(1.5794302666547846,2.6985789450702233,0) q[11];
u3(2.8670460259051826,-1.4210661597787428,0) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
u3(1.9787438939980078,1.8904301876881417,0) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
u3(2.28399350890765,-2.982001116207003,-pi) q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
u3(2.1761633245663865,1.904198228238804,0) q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
u3(0.27078867528550604,1.5210075835580792,0) q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
u3(0.7545152110842547,0.3617744290191398,-pi) q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
u3(1.4876032641952892,0.41159274487845465,-pi) q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
u3(0.765941383327034,-0.15624869766433136,-pi) q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
u3(2.6232873181840106,1.8397039425941601,0) q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
u3(2.4782292522231346,-2.7378913330613686,-pi) q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
u3(2.526866864605136,-1.0071453217107083,-pi) q[10];
u3(1.6475495893366514,3.110482283900007,0) q[11];
u3(3.0569793381207737,3.008509421420701,-pi) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];

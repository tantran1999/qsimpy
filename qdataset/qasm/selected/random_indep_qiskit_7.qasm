// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
tdg q[0];
u1(2.4248043857342108) q[1];
u2(pi/4,-pi) q[2];
u2(0,0) q[3];
cx q[3],q[1];
ry(-1.6110232157240438) q[1];
ry(-1.6110232157240438) q[3];
cx q[3],q[1];
u2(pi/4,0.7167882678555819) q[1];
cx q[0],q[1];
tdg q[1];
u2(-pi,-pi) q[3];
u3(1.7745821253041683,-2.4764232673109765,-1.023742495249473) q[4];
cx q[6],q[2];
tdg q[2];
cx q[5],q[2];
t q[2];
h q[5];
cx q[3],q[5];
h q[5];
cu1(pi/2) q[3],q[5];
u2(0,0) q[3];
u1(pi/2) q[5];
cx q[6],q[2];
u2(0,3*pi/4) q[2];
cx q[2],q[1];
t q[1];
cx q[0],q[1];
u1(3.2536483225524107) q[0];
crx(5.2625683110417505) q[0],q[5];
u2(1.7603577140039537,3*pi/4) q[1];
p(2.026167409578137) q[2];
cx q[3],q[1];
ry(-0.9623259560190263) q[1];
ry(0.9623259560190263) q[3];
cx q[3],q[1];
u1(-1.7603577140039537) q[1];
u2(3*pi/4,-pi) q[3];
cu3(1.5640163777242566,2.405358320221107,4.07828399178064) q[4],q[1];
p(4.999421715326975) q[4];
cu3(6.202284721500185,5.076820509783505,5.29650934899458) q[4],q[3];
u3(pi,-pi/2,pi/2) q[3];
u1(2.313248925660014) q[4];
cx q[5],q[1];
cx q[1],q[5];
u3(pi,pi/8,-pi/8) q[5];
u2(-pi/2,-pi/2) q[6];
crz(4.875952527358193) q[6],q[2];
cx q[0],q[2];
cx q[2],q[0];
u3(1.8272198079864164,-1.2096331269843725,1.450852124720936) q[0];
ch q[1],q[2];
tdg q[1];
z q[2];
u3(1.6408049939899045,0.7175293395407141,0.028344810415217747) q[6];
cx q[6],q[4];
ry(-2.5658469779639455) q[4];
ry(2.5658469779639455) q[6];
cx q[6],q[4];
u2(-1.986547039963356,0.9767093236508302) q[4];
cu1(pi/2) q[2],q[4];
u2(-pi/2,pi/2) q[4];
u2(-pi,pi/2) q[6];
rzz(2.917738905811407) q[6],q[3];
u2(0.25455381073759664,2.4109111940327272) q[3];
rx(-pi/2) q[6];
cswap q[6],q[1],q[5];
s q[1];
cx q[1],q[4];
x q[1];
cy q[4],q[1];
u2(1.4683194020151475,3.4154613287486493) q[1];
x q[4];
s q[5];
sx q[6];
cx q[5],q[6];
cx q[2],q[5];
x q[2];
ch q[5],q[1];
u1(0.4199696152223926) q[1];
h q[5];
cx q[2],q[5];
rz(6.02771839731381) q[5];
cx q[2],q[5];
u2(pi/2,-pi) q[5];
ccx q[6],q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
rxx(0.6965106857196179) q[0],q[3];
u2(-2.9109148838774797,pi/2) q[0];
u2(0,0) q[3];
cx q[3],q[1];
ry(-1.366386517363651) q[1];
ry(1.366386517363651) q[3];
cx q[3],q[1];
u3(2.9766777365729253,-pi/2,1.1508267115725044) q[1];
u2(-pi,pi/2) q[3];
rzz(0.16462968369296951) q[3],q[0];
rx(-pi/2) q[0];
rx(-pi/2) q[3];
x q[6];
cu3(2.510079506040741,2.342819928892192,2.695349594570543) q[6],q[4];
h q[4];
cu1(pi/2) q[6],q[4];
u2(-pi/2,pi/2) q[4];
cx q[4],q[5];
cx q[5],q[4];
h q[5];
cry(0.5915207724126751) q[6],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];

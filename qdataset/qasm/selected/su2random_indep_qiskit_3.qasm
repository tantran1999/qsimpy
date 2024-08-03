// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];
u3(1.4368347742816616,1.563280899135842,-pi) q[0];
u3(0.13038834331032634,3.1322119352256292,0) q[1];
cx q[0],q[1];
u3(2.301856027570534,-1.7291536732871133,-pi) q[2];
cx q[0],q[2];
u3(1.2444656817555668,0.5550554224571163,0) q[0];
cx q[1],q[2];
u3(1.5046299106322627,1.1646500873100205,-pi) q[1];
cx q[0],q[1];
u3(1.0625547235745711,-0.2928382424047804,0) q[2];
cx q[0],q[2];
u3(0.024807688980383977,-2.434570523812673,0) q[0];
cx q[1],q[2];
u3(3.0649864034230183,1.3933297522764283,-pi) q[1];
cx q[0],q[1];
u3(1.1773372206208805,-1.3076812305427237,-pi) q[2];
cx q[0],q[2];
u3(0.5166404252966228,-2.248311899378857,-pi) q[0];
cx q[1],q[2];
u3(1.793373244068875,-0.7958234754631421,-pi) q[1];
u3(2.8742785055981956,1.0941137716709264,-pi) q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];

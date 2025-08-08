OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg reg_measure[0];
barrier q[0],q[1],q[2],q[3],q[4];

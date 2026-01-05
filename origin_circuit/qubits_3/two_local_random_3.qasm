OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg reg_measure[0];
barrier q[0],q[1],q[2];

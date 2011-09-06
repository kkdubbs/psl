/*
 * This file is part of the PSL software.
 * Copyright 2011 University of Maryland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.umd.cs.psl.optimizer.conic.program;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.umd.cs.psl.optimizer.conic.program.graph.Node;
import edu.umd.cs.psl.optimizer.conic.program.graph.Relationship;

public class SecondOrderCone extends Cone {
	SecondOrderCone(ConicProgram p, int n) {
		super(p);
		if (n < 2)
			throw new IllegalArgumentException("Second-order cones must have at least two dimensions.");
		Variable var = null;
		for (int i = 0; i < n; i++) {
			var = new Variable(p);
			node.createRelationship(ConicProgram.CONE_REL, var.getNode());
		}
		node.createRelationship(ConicProgram.SOC_N_REL, var.getNode());
		var.setValue(1.5 * (n-1));
		var.setDualValue(1.5 * (n-1));
		p.notify(ConicProgramEvent.SOCCreated, this);
	}
	
	SecondOrderCone(ConicProgram p, Node n) {
		super(p, n);
	}
	
	@Override
	NodeType getType() {
		return NodeType.soc;
	}
	
	public int getN() {
		int n = 0;
		Iterator<Relationship> rels = node.getRelationshipIterator(ConicProgram.CONE_REL);
		while (rels.hasNext()) {
			rels.next();
			n++;
		}
		return n;
	}
	
	public Set<Variable> getVariables() {
		Set<Variable> vars = new HashSet<Variable>();
		for (Relationship r : node.getRelationships(ConicProgram.CONE_REL))
			vars.add((Variable) Entity.createEntity(program, r.getEnd()));
		return vars;
	}
	
	public Variable getNthVariable() {
		return (Variable) Entity.createEntity(program, node.getRelationshipIterator(ConicProgram.SOC_N_REL).next().getEnd());
	}
	
	@Override
	public final void delete() {
		program.verifyCheckedIn();
		program.notify(ConicProgramEvent.SOCDeleted, this);
		node.getRelationshipIterator(ConicProgram.SOC_N_REL).next().delete();
		for (Variable v : getVariables()) {
			v.getNode().getRelationshipIterator(ConicProgram.CONE_REL).next().delete();
			v.delete();
		}
		super.delete();
	}
	
	public void setBarrierGradient(Map<Variable, Integer> varMap, DoubleMatrix1D x, DoubleMatrix1D g) {
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices);
		DoubleMatrix1D gSel = g.viewSelection(indices);
		double coeff = (double) 2 / (Math.pow(xSel.get(i), 2) - xSel.zDotProduct(xSel, 0, i));
		gSel.assign(xSel).assign(DoubleFunctions.mult(coeff));
		gSel.set(i, gSel.get(i) * -1);
	}
	
	void setBarrierHessian(Map<Variable, Integer> varMap, DoubleMatrix1D x, DoubleMatrix2D H) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices).copy();
		xSel.set(i, xSel.get(i)*-1);
		DoubleMatrix2D HSel = H.viewSelection(indices, indices);
		double coeff = (double) 2 / (Math.pow(xSel.get(i), 2) - xSel.zDotProduct(xSel, 0, i));
		HSel.assign(alg.multOuter(xSel, xSel, null).assign(DoubleFunctions.mult(Math.pow(coeff, 2))));
		for (int j = 0; j < i; j++)
			HSel.set(j, j, HSel.get(j, j) + coeff);
		HSel.set(i, i, HSel.get(i,i) - coeff);
	}
	
	public void setBarrierHessianInv(Map<Variable, Integer> varMap, DoubleMatrix1D x, DoubleMatrix2D Hinv) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices).copy();
		xSel.set(i, xSel.get(i)*-1);
		DoubleMatrix2D HSel = Hinv.viewSelection(indices, indices);
		double coeff = (double) 2 / (Math.pow(xSel.get(i), 2) - xSel.zDotProduct(xSel, 0, i));
		HSel.assign(alg.multOuter(xSel, xSel, null).assign(DoubleFunctions.mult(Math.pow(coeff, 2))));
		for (int j = 0; j < i; j++)
			HSel.set(j, j, HSel.get(j, j) + coeff);
		HSel.set(i, i, HSel.get(i,i) - coeff);
		try{
		HSel.assign(alg.inverse(HSel));
		}
		catch (IllegalArgumentException e) {
			System.out.println(x.viewSelection(indices));
		System.out.println(alg.toVerboseString(HSel));
		throw e;
		}
	}

	@Override
	boolean isInterior(Map<Variable, Integer> varMap, DoubleMatrix1D x) {
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices);
		return xSel.get(i) > Math.sqrt(xSel.zDotProduct(xSel, 0, i)) + 0.05;
	}

	@Override
	void setInteriorDirection(Map<Variable, Integer> varMap, DoubleMatrix1D x,
			DoubleMatrix1D d) {
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices);
		DoubleMatrix1D dSel = d.viewSelection(indices);
		dSel.assign(0.0);
		if (xSel.get(i) <= Math.sqrt(xSel.zDotProduct(xSel, 0, i)) + 0.05)
			dSel.set(i, Math.sqrt(xSel.zDotProduct(xSel, 0, i)) + 0.25 - xSel.get(i));
	}

	@Override
	public double getMaxStep(Map<Variable, Integer> varMap, DoubleMatrix1D x,
			DoubleMatrix1D dx) {
		Set<Variable> vars = getVariables();
		int[] indices = new int[vars.size()];
		Variable varN = getNthVariable();
		vars.remove(varN);
		int i = 0;
		for (Variable v : vars) {
			indices[i++] = varMap.get(v);
		}
		indices[i] = varMap.get(varN);
		DoubleMatrix1D xSel = x.viewSelection(indices);
		DoubleMatrix1D dxSel = dx.viewSelection(indices);
		
		double a = Math.pow(dxSel.get(i), 2) - dxSel.zDotProduct(dxSel, 0, i);
		double b = 2*(dxSel.get(i)*xSel.get(i) - dxSel.zDotProduct(xSel, 0, i));
		double c = Math.pow(xSel.get(i), 2) - xSel.zDotProduct(xSel, 0, i);
		
		double sol1 = (-1 * b + Math.sqrt(Math.pow(b, 2) - 4 * a * c)) / (2 * a);
		double sol2 = (-1 * b - Math.sqrt(Math.pow(b, 2) - 4 * a * c)) / (2 * a);
		
		if (!Double.isNaN(sol1) && !Double.isNaN(sol2)) {
			if (sol1 > 0 && sol2 > 0)
				return Math.min(sol1, sol2) * .95;
			else if (sol1 > 0)
				return sol1 * .95;
			else if (sol2 > 0)
				return sol2 * .95;
			else
				return Double.MAX_VALUE;
		}
		else if (!Double.isNaN(sol1))
			if (sol1 > 0)
				return sol1;
			else
				return 1.0;
		else if (!Double.isNaN(sol2))
			if (sol2 > 0)
				return sol2;
			else
				return 1.0;
		else throw new IllegalStateException();
	}
}


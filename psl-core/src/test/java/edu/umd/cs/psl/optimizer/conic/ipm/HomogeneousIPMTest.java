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
package edu.umd.cs.psl.optimizer.conic.ipm;

import java.util.List;
import java.util.Vector;

import edu.umd.cs.psl.config.EmptyBundle;
import edu.umd.cs.psl.optimizer.conic.ConicProgramSolver;
import edu.umd.cs.psl.optimizer.conic.ConicProgramSolverContractTest;

public class HomogeneousIPMTest extends ConicProgramSolverContractTest {

	@Override
	protected List<? extends ConicProgramSolver> getConicProgramSolverImplementations() {
		Vector<HomogeneousIPM> solvers = new Vector<HomogeneousIPM>(1);
		solvers.add(new HomogeneousIPM(new EmptyBundle()));
		return solvers;
	}

}

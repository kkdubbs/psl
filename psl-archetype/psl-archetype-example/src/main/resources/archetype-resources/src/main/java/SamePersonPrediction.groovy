#set( $symbol_pound = '#' )
#set( $symbol_dollar = '$' )
#set( $symbol_escape = '\' )

/*
* This file is part of the PSL software.
* Copyright 2011-2015 University of Maryland
* Copyright 2013-2015 The Regents of the University of California
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

package ${package};

import java.lang.annotation.*;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.Iterator;

import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight

/*
 * In this example program, the task is to align two social networks, by
 * identifying which pairs of users are the same across networks.
 */

class MyStringSimilarity implements ExternalFunction {
	
	
	public int getArity() {
		return 2;
	}

	
	public ArgumentType[] getArgumentTypes() {
		return [ArgumentType.String, ArgumentType.String].toArray();
	}
	
	
	public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
		return args[0].toString().equals(args[1].toString()) ? 1.0 : 0.0;
	}
	
}


class SamePersonPrediction {

	class SamePersonPredictionExperiment{
		public ConfigManager cm;
		public ConfigBundle cb;

		public double initialWeight;

		public boolean doWeightLearning;
		public boolean createNewDataStore;

		public String dataDirectory; 

		Set<Predicate> closedTest;
		Set<Predicate> closedTrain;

		Map<Predicate,String> socialNetwork1;
		Map<Predicate,String> socialNetwork2;
		Map<Predicate,String> truthFileMap;

	}


	def setupConfig(){
    	SamePersonPredictionExperiment config = new SamePersonPredictionExperiment();
    	config.cm = ConfigManager.getManager();
    	config.cb = config.cm.getBundle("basic-example");
    	config.initialWeight = config.cb.getDouble("initialRuleWeight", 1.0);
		config.doWeightLearning = config.cb.getBoolean("doWeightLearning", true);
    	config.createNewDataStore = config.cb.getBoolean("createNewDataStore", true);
    	config.dataDirectory = 'data'+java.io.File.separator+'sn'+java.io.File.separator;

    	
    	return config;

    }

    def setupDataStore(config){
    	String dbpath = "./testdb_basic_example";
    	DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, config.createNewDataStore), config.cb);

    	return data;
    }

    def loadPredicateFileMaps(config){
    	config.socialNetwork1 = [((Predicate)Network):"sn_network.txt",
	                 ((Predicate)Knows):"sn_knows.txt"];

	    config.socialNetwork2 = [((Predicate)Name):"sn2_names.txt",
	                 ((Predicate)Network):"sn2_network.txt",
	                 ((Predicate)Knows):"sn2_knows.txt"];

	    config.truthFileMap = [((Predicate)SamePerson):"sn_align.txt"];

    }

    def definePredicates(config, data, m){
    	/* 
		 * We create four predicates in the model, giving their names and list of argument types
		 */

		m.add predicate: "Network",    types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "Name",       types: [ArgumentType.UniqueID, ArgumentType.String]
		m.add predicate: "Knows",      types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "SamePerson", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

		/*
		 * Now, we define a string similarity function bound to a predicate.
		 * Note that we can use any implementation of ExternalFunction that acts on two strings!
		 */

		m.add function: "SameName" , implementation: new LevenshteinSimilarity()

		/* Also, try: new MyStringSimilarity(), see end of file */

		/* Finally, we define our set of closed predicates for train and test */
		config.closedTest = [Network, Name, Knows] as Set;
		config.closedTrain = [Network, Name, Knows, SamePerson] as Set;

    }

    def defineModel(config, data, m){
		/* 
		 * Having added all the predicates we need to represent our problem, we finally
		 * add some rules into the model. Rules are defined using a logical syntax.
		 * 
		 * Uppercase letters are variables and the predicates used in the rules below
		 * are those defined above. The character '&' denotes a conjunction where '>>'
		 * denotes an implication.
		 * 
		 * Each rule is given a weight that is either the weight used for inference or
		 * an initial guess for the starting point of weight learning.
		 */

		/*
		 * We also create constants to refer to each social network.
		 */

		GroundTerm snA = data.getUniqueID(1);
		GroundTerm snB = data.getUniqueID(2);

		/*
		 * Our first rule says that users with similar names are likely the same person
		 */
		m.add rule : ( Network(A, snA) & Network(B, snB) & Name(A,X) & Name(B,Y)
			& SameName(X,Y) ) >> SamePerson(A,B),  weight : 5

		/* 
		 * In this rule, we use the social network to propagate SamePerson information.
		 */
		m.add rule : ( Network(A, snA) & Network(B, snB) & SamePerson(A,B) & Knows(A, Friend1)
			& Knows(B, Friend2) ) >> SamePerson(Friend1, Friend2) , weight : 3.2

		/* 
		 * Next, we define some constraints for our model. In this case, we restrict that
		 * each person can be aligned to at most one other person in the other social network.
		 * To do so, we define two partial functional constraints where the latter is on
		 * the inverse. Please read more about PartialFunctional constraints on our wiki.
		 We also say that samePerson must be symmetric,
		 * i.e., samePerson(p1, p2) == samePerson(p2, p1).
		 */
		m.add PredicateConstraint.PartialFunctional, on : SamePerson
		m.add PredicateConstraint.PartialInverseFunctional, on : SamePerson
		m.add PredicateConstraint.Symmetric, on : SamePerson

		/*
		 * Finally, we define a prior on the inference predicate samePerson. It says that
		 * we should assume two people are not the samePerson with some weight. This can
		 * be overridden with evidence as defined in the previous rules.
		 */
		m.add rule: ~SamePerson(A,B), weight: 1

    }

    def loadEvidenceFromValues(data, evidencePartition){

    	def insert = data.getInserter(Name, evidencePartition);

    	/* Social Network A */
		insert.insert(1, "John Braker");
		insert.insert(2, "Mr. Jack Ressing");
		insert.insert(3, "Peter Larry Smith");
		insert.insert(4, "Tim Barosso");
		insert.insert(5, "Jessica Pannillo");
		insert.insert(6, "Peter Smithsonian");
		insert.insert(7, "Miranda Parker");

		/* Social Network B */
		insert.insert(11, "Johny Braker");
		insert.insert(12, "Jack Ressing");
		insert.insert(13, "PL S.");
		insert.insert(14, "Tim Barosso");
		insert.insert(15, "J. Panelo");
		insert.insert(16, "Gustav Heinrich Gans");
		insert.insert(17, "Otto v. Lautern");


    }

    def loadPredicateFromFile(data, config, evidencePartition, predicateFileMap){

    	/*
		 * Of course, we can also load data directly from tab delimited data files.
		 */

		def dir = config.dataDirectory;

		for (Predicate p : predicateFileMap.keySet()){
			def inserter = data.getInserter(p, evidencePartition);
			def filename = predicateFileMap[p];

			InserterUtils.loadDelimitedData(inserter, dir+filename);
		}

    }

    def loadPredicateFromFileWithTruth(data, config, evidencePartition, predicateFileMap){

    	/*
		 * we can also load data directly from tab delimited data files with truth values specified.
		 */

		def dir = config.dataDirectory;

		for (Predicate p : predicateFileMap.keySet()){
			def inserter = data.getInserter(p, evidencePartition);
			def filename = predicateFileMap[p];

			InserterUtils.loadDelimitedDataTruth(inserter, dir+filename);
		}
		
    }

    def populateDB(data, startIndex1, endIndex1, startIndex2, endIndex2, db){
		/*
		 * Before running inference, we have to add the target atoms to the database.
		 * If inference (or learning) attempts to access an atom that is not in the database,
		 * it will throw an exception.
		 * 
		 * The below code builds a set of all users, then uses a utility class
		 * (DatabasePopulator) to create all possible SamePerson atoms between users of
		 * each network.
		 */

		def popMap = getVariablePopulationMap(data, startIndex1, endIndex1, startIndex2, endIndex2);

		DatabasePopulator dbPop = new DatabasePopulator(db);
		dbPop.populate((SamePerson(UserA, UserB)).getFormula(), popMap);
		dbPop.populate((SamePerson(UserB, UserA)).getFormula(), popMap);

    }

    def runInference(m, data, config, targetPartition, evidencePartition, start1, end1, start2, end2){

    	Database db = data.getDatabase(targetPartition, config.closedTest, evidencePartition);
    	populateDB(data, start1, start2, end1, end2, db);

    	/*
		 * Now we can run inference
		 */

		MPEInference inferenceApp = new MPEInference(m, db, config.cb);
		inferenceApp.mpeInference();

		/*
		 * Let's see the results
		 */
		println "Inference results with hand-defined weights:"
		DecimalFormat formatter = new DecimalFormat("${symbol_pound}.${symbol_pound}${symbol_pound}");
		for (GroundAtom atom : Queries.getAllAtoms(db, SamePerson))
			println atom.toString() + "${symbol_escape}t" + formatter.format(atom.getValue());

		inferenceApp.close();
		db.close();

    }

    def getVariablePopulationMap(data, startIndex1, endIndex1, startIndex2, endIndex2){
    	Set<GroundTerm> usersA = new HashSet<GroundTerm>();
		Set<GroundTerm> usersB = new HashSet<GroundTerm>();
		for (int i = startIndex1; i < endIndex1; i++)
			usersA.add(data.getUniqueID(i));
		for (int i = startIndex2; i < endIndex2; i++)
			usersB.add(data.getUniqueID(i));

		Map<Variable, Set<GroundTerm>> popMap = new HashMap<Variable, Set<GroundTerm>>();
		popMap.put(new Variable("UserA"), usersA)
		popMap.put(new Variable("UserB"), usersB)

		return popMap;
    }

    def learnWeights(m, data, config, truthPartition, evidencePartition, targetPartition, start1, end1, start2, end2){
    	/* 
		 * Now, we can learn the weights.
		 * 
		 * We first open a database which contains all the target atoms as observations.
		 * We then combine this database with the original database to learn.
		 */

		Database distributionDB = data.getDatabase(targetPartition, config.closedTest, evidencePartition);
		Database trueDataDB = data.getDatabase(truthPartition, config.closedTrain);

		populateDB(data, start1, start2, end1, end2, distributionDB);

		MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, distributionDB, trueDataDB, config.cb);
		weightLearning.learn();
		weightLearning.close();

		distributionDB.close();
		trueDataDB.close();

	   	/*
		 * Let's have a look at the newly learned weights.
		 */

		println ""
		println "Learned model:"
		println m
    }

    static void main(String[] args){
    	def spp = new SamePersonPrediction();
    	def config = spp.setupConfig();
	    def data = spp.setupDataStore(config);
	    PSLModel m = new PSLModel(spp, data);

	    spp.definePredicates(config, data, m);
	    spp.defineModel(config, data, m);

	    /*Let's first load evidence directly for our first social network*/
	    Partition evidencePartition = data.getPartition("evidencePartition");
	    spp.loadEvidenceFromValues(data, evidencePartition);

	    /*Let's load data from file now. First, we set up our predicate --- filename map*/
	    spp.loadPredicateFileMaps(config);
	    
	    spp.loadPredicateFromFile(data, config, evidencePartition, config.socialNetwork1);

	    /*Let's see what happens when we run inference. First, we'll need an empty write partition*/
	    Partition targetPartition = data.getPartition("targetPartition");
	    spp.runInference(m, data, config, targetPartition, evidencePartition, 1, 8, 11, 18);


	    /*Now let's try to learn the rule weights from training data
	    *First we'll need to load the truth labels for training */
	    Partition truthPartition = data.getPartition("truthPartition");
	    spp.loadPredicateFromFileWithTruth(data, config, truthPartition, config.truthFileMap);

	    spp.learnWeights(m, data, config, truthPartition, evidencePartition, targetPartition, 1, 8, 11, 18);


	    /*Now let's load a new social network and run inference again!*/
	    Partition evidencePartition2 = data.getPartition("evidencePartition2");
	    Partition targetPartition2 = data.getPartition("targetPartition2");

	    spp.loadPredicateFromFile(data, config, evidencePartition2, config.socialNetwork2);

	    spp.runInference(m, data, config, targetPartition2, evidencePartition2, 21, 28, 31, 38);
    }


}
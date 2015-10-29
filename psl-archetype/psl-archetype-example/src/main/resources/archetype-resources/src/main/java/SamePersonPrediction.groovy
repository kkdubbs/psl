#set( $symbol_pound = '#' )
#set( $symbol_dollar = '$' )
#set( $symbol_escape = '\' )

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

		Set<Partition> closedTest;
		Set<Partition> closedTrain;

		public Partition trainRead;
		public Partition trainWrite;
		public Partition trainLabels;

		public Partition testRead;
		public Partition testWrite;
		public Partition testLabels;

	}


	def setupConfig(){
    	SamePersonPredictionExperiment config = new SamePersonPredictionExperiment();
    	config.cm = ConfigManager.getManager();
    	config.cb = config.cm.getBundle("basic-example");
    	config.initialWeight = 5.0;
		config.doWeightLearning = true;
    	config.createNewDataStore = true;
    	config.dataDirectory = 'data'+java.io.File.separator+'sn'+java.io.File.separator;

    	
    	return config;

    }

    def setupDataStore(config){
    	String dbpath = "./testdb_basic_example";
    	DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, config.createNewDataStore), config.cb);

    	return data;
    }


    def defineModel(config, data, m){
		/*
		 * In this example program, the task is to align two social networks, by
		 * identifying which pairs of users are the same across networks.
		 */

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
		 * the inverse. We also say that samePerson must be symmetric,
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

		config.closedTest = [Network, Name, Knows] as Set;
		config.closedTrain = [Network, Name, Knows] as Set;

    }

    def loadData(data, config){
		 /* 
		 * We now insert data into our DataStore. All data is stored in a partition.
		 * We put all the observations into their own partition.
		 * 
		 * We can use insertion helpers for a specified predicate. Here we show how one
		 * can manually insert data or use the insertion helpers to easily implement
		 * custom data loaders.
		 */
		Partition evidencePartition = data.getPartition("read_evidence");
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

		/*
		 * Of course, we can also load data directly from tab delimited data files.
		 */
		def dir = config.dataDirectory;

		insert = data.getInserter(Network, evidencePartition)
		InserterUtils.loadDelimitedData(insert, dir+"sn_network.txt");

		insert = data.getInserter(Knows, evidencePartition)
		InserterUtils.loadDelimitedData(insert, dir+"sn_knows.txt");

		config.testRead = evidencePartition;
		config.trainRead = evidencePartition;

		/* 
		* Later, we'll be using labeled training data to learn weights 
		* so we'll load these now
		*/

		Partition trainingLabels = data.getPartition("train_labels");
		insert = data.getInserter(SamePerson, trainingLabels);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "sn_align.txt");
		config.trainLabels = trainingLabels;

		config.testWrite = data.getPartition("test_write");
		config.trainWrite = data.getPartition("train_write");
    }

    def updateInferenceData(data, config){

    	data.deletePartition(data.getPartition("read_evidence"));
		data.deletePartition(config.testWrite);
		
		Partition evidencePartition = data.getPartition("test_evidence");
		def dir = config.dataDirectory;

		def insert = data.getInserter(Network, evidencePartition)
		InserterUtils.loadDelimitedData(insert, dir+"sn2_network.txt");

		insert = data.getInserter(Name, evidencePartition);
		InserterUtils.loadDelimitedData(insert, dir+"sn2_names.txt");

		insert = data.getInserter(Knows, evidencePartition);
		InserterUtils.loadDelimitedData(insert, dir+"sn2_knows.txt");

		config.testWrite = data.getPartition("test_write");
		config.testRead = evidencePartition;

    }

    def runInference(m, data, config, startIndex1, endIndex1, startIndex2, endIndex2){
    	
		def targetPartition = config.testWrite;
		def evidencePartition = config.testRead;
		Database db = data.getDatabase(targetPartition, config.closedTest, evidencePartition);

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

		/*
		 * Now we can run inference
		 */

		MPEInference inferenceApp = new MPEInference(m, db, config.cb);
		inferenceApp.mpeInference();
		inferenceApp.close();



		/*
		 * Let's see the results
		 */
		println "Inference results with hand-defined weights:"
		DecimalFormat formatter = new DecimalFormat("${symbol_pound}.${symbol_pound}${symbol_pound}");
		for (GroundAtom atom : Queries.getAllAtoms(db, SamePerson))
			println atom.toString() + "${symbol_escape}t" + formatter.format(atom.getValue());

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

    def learnWeights(m, data, config, startIndex1, endIndex1, startIndex2, endIndex2){
    	/* 
		 * Now, we can learn the weights.
		 * 
		 * We first open a database which contains all the target atoms as observations.
		 * We then combine this database with the original database to learn.
		 */

		def trueDataPartition = config.trainLabels;
		def trainTargetPartition = config.trainWrite;
		def evidencePartition = config.trainRead;

		Database distributionDB = data.getDatabase(trainTargetPartition, config.closedTrain, evidencePartition);
		Database trueDataDB = data.getDatabase(trueDataPartition, [samePerson] as Set);

		def popMap = getVariablePopulationMap(data, startIndex1, endIndex1, startIndex2, endIndex2);

		DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
		dbPop.populate((SamePerson(UserA, UserB)).getFormula(), popMap);
		dbPop.populate((SamePerson(UserB, UserA)).getFormula(), popMap);

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

	    spp.defineModel(config, data, m);
	    spp.loadData(data, config);
	    spp.runInference(m, data, config, 1, 8, 11, 18);
	    spp.learnWeights(m, data, config, 1, 8, 11, 18);

	    spp.updateInferenceData(data, config);
	    spp.runInference(m, data, config, 21, 28, 31, 38);

    }


}
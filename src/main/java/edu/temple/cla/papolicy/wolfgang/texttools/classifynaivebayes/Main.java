/*
 * Copyright (c) 2018, Temple University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * All advertising materials features or use of this software must display 
 *   the following  acknowledgement
 *   This product includes software developed by Temple University
 * * Neither the name of the copyright holder nor the names of its 
 *   contributors may be used to endorse or promote products derived 
 *   from this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
package edu.temple.cla.papolicy.wolfgang.texttools.classifynaivebayes;

import edu.temple.cla.papolicy.wolfgang.texttools.util.CommonFrontEnd;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Util;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Vocabulary;
import edu.temple.cla.papolicy.wolfgang.texttools.util.WordCounter;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import picocli.CommandLine;

/**
 * Program to classify text based on the Naive Bayes algorithm. 
 * This a Naive Bayes classifier written from scratch that attempts to produce
 * comparable results to <a href="http://mallet.cs.umass.edu/">MALLET</a>. In
 * testing with the Bills_Data using the 2001-2013 sessions as training set and
 * 2015 session as test set, this classifier achieves 72% accuracy while MALLET
 * achieved 75%. It agrees with the MALLET classifier 82%. Of the 18%
 * disagreements 4% where the correct result.
 *
 * The code is based on
 * <a href="http://blog.datumbox.com/machine-learning-tutorial-the-naive-bayes-text-classifier/">
 * Machine Learning Tutorial: The Native Bayes Text Classifier </a> and
 * examination of the MALLET code.
 *
 * @author Paul Wolfgang
 */
public class Main implements Callable<Void> {

    @CommandLine.Option(names = "--output_table_name",
            description = "Table where results are written")
    private String outputTableName;

    @CommandLine.Option(names = "--output_code_col", required = true,
            description = "Column where the result is set")
    private String outputCodeCol;

    @CommandLine.Option(names = "--model",
            description = "Directory where model files are written")
    private String modelDir = "Model_Dir";

    private final String[] args;

    public Main(String[] args) {
        this.args = args;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Main main = new Main(args);
        CommandLine commandLine = new CommandLine(main);
        commandLine.setUnmatchedArgumentsAllowed(true).parse(args);
        try {
            main.call();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    @Override
    public Void call() {
        List<String> ids = new ArrayList<>();
        List<String> ref = new ArrayList<>();
        List<WordCounter> counts = new ArrayList<>();
        List<Integer> cats = new ArrayList<>();
        Vocabulary problemVocab = new Vocabulary(); // Not used
        CommonFrontEnd commonFrontEnd = new CommonFrontEnd();
        CommandLine commandLine = new CommandLine(commonFrontEnd);
        commandLine.setUnmatchedArgumentsAllowed(true);
        commandLine.parse(args);
        commonFrontEnd.loadData(ids, ref, problemVocab, counts);
        File modelParent = new File(modelDir);
        Map<String, Double> prior
                = (Map<String, Double>) Util.readFile(modelParent, "prior.bin");
        Map<String, Map<String, Double>> condProb
                = (Map<String, Map<String, Double>>) Util.readFile(modelParent, "condProp.bin");
        List<Integer> categories = new ArrayList<>();
        for (int i = 0; i < counts.size(); i++) {
            Integer cat = classify(counts.get(i), prior, condProb);
            categories.add(cat);
        }
        String outputTable = outputTableName != null ? outputTableName : commonFrontEnd.getTableName();
        if (outputCodeCol != null) {
            System.err.println("Inserting result into database");
            commonFrontEnd.outputToDatabase(outputTable,
                    outputCodeCol,
                    ids,
                    categories);
        }
        System.err.println("SUCESSFUL COMPLETION");

        return null;
    }


    public Integer classify(WordCounter counter, Map<String, Double> prior, Map<String, Map<String, Double>> condProb) {
        SortedMap<Double, String> testCats = new TreeMap<>();
        prior.forEach((cat, priorProb) -> {
            double score = Math.log(priorProb);
            score += counter.getWords().stream()
                    .collect(Collectors.summingDouble(word -> {
                        int countOfWord = counter.getCount(word);
                        if (condProb.containsKey(word)) {
                            double term = Math.log(condProb.get(word).get(cat)) * countOfWord;
                            return term;
                        } else {
                            return 0;
                        }
                    }));
            testCats.put(score, cat);
        });
        String winningCat = testCats.get(testCats.lastKey());
        return new Integer(winningCat);
    }
}

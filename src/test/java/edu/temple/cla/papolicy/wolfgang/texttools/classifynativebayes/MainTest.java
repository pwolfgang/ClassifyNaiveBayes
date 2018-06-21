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
package edu.temple.cla.papolicy.wolfgang.texttools.classifynativebayes;

import edu.temple.cla.papolicy.wolfgang.texttools.classifynaivebayes.Main;
import edu.temple.cla.papolicy.wolfgang.texttools.util.CommonFrontEnd;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Vocabulary;
import edu.temple.cla.papolicy.wolfgang.texttools.util.WordCounter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.Before;
import org.junit.Test;
import picocli.CommandLine;

/**
 *
 * @author Paul
 */
public class MainTest {
    
    List<String> ref = new ArrayList<>();
    List<String> ids = new ArrayList<>();
    Vocabulary vocabulary = new Vocabulary();
    List<WordCounter> counts = new ArrayList<>();
    String[] args = {"--datasource", "TestDb.txt",
                "--table_name", "TestTableUnknown",
                "--id_column", "ID",
                "--text_column", "Abstract",
                "--code_column", "Code",
                "--remove_stopwords", "false",
                "--output_code_col", "Code"};
    Map<String, Integer> docsInTrainingSet = new TreeMap<>();
    Map<String, WordCounter> trainingSets = new TreeMap<>();
    Map<String, Double> prior = new TreeMap<>();
    Map<String, Map<String, Double>> condProb = new TreeMap<>();

    public MainTest() {
    }

    @Before
    public void Setup() {
        TestDatabase.createTestTable();
    }
    
    
    
    @Test
    public void testMain() {
        Main.main(args);
    }

    
}

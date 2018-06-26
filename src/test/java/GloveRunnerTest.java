import lombok.extern.slf4j.Slf4j;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.stream.DoubleStream;

class GloveRunner {


	public WeightLookupTable<VocabWord> run() throws IOException {

			File inputFile = new ClassPathResource("raw_sentences.txt").getFile();

			// creating SentenceIterator wrapping our training corpus
			SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

			// Split on white spaces in the line to get words
			TokenizerFactory t = new DefaultTokenizerFactory();
			t.setTokenPreProcessor(new CommonPreprocessor());

			Glove glove = new Glove.Builder().iterate(iter).tokenizerFactory(t)

					.alpha(0.75).learningRate(0.1)
					.epochs(25)                        // number of epochs for training
					.xMax(100)                         // cutoff for weighting function
					.batchSize(1000)                   // training is done in batches taken from training corpus
					.shuffle(true)                     // if set to true, batches will be shuffled before training
					.symmetric(true).build();          // if set to true word pairs will be built in both directions, LTR and RTL

			glove.fit();
			return glove.getLookupTable();
		}
	}

@Slf4j
public class GloveRunnerTest {

	@Test
	public void test() throws IOException {
		for (int i = 0; i < 10; i++) {
			log.info("Running Glove. Iteration {}", i);
			WeightLookupTable<VocabWord> weightsLookupTable = new GloveRunner().run();

			INDArray weights = weightsLookupTable.getWeights();
			double[] doubles = weights.data().asDouble();
			boolean containsNaN = DoubleStream.of(doubles).filter(Double::isNaN).findAny().isPresent();
			Assert.assertFalse(String.format("Iteration %d: WeightLookupTable contains NaN instances!", i), containsNaN);
		}

		System.exit(0);
	}

}

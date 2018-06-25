package my.bug;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;


public class GloveRunner {


	public WeightLookupTable<VocabWord> run() {

		try {
			File inputFile = new ClassPathResource("my/bug/raw_sentences.txt").getFile();

			// creating SentenceIterator wrapping our training corpus
			SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

			// Split on white spaces in the line to get words
			TokenizerFactory t = new DefaultTokenizerFactory();
			t.setTokenPreProcessor(new CommonPreprocessor());

			Glove glove = new Glove.Builder().iterate(iter).tokenizerFactory(t)


					.alpha(0.75).learningRate(0.1)

					// number of epochs for training
					.epochs(25)

					// cutoff for weighting function
					.xMax(100)

					// training is done in batches taken from training corpus
					.batchSize(1000)

					// if set to true, batches will be shuffled before training
					.shuffle(true)

					// if set to true word pairs will be built in both directions, LTR and RTL
					.symmetric(true).build();

			glove.fit();
			return glove.getLookupTable();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

}

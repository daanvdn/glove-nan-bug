package my.bug;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.stream.DoubleStream;

@Slf4j
public class GloveRunnerTest {

	@Test
	public void test() {
		for (int i = 0; i < 10; i++) {
			log.info("Running Glove. Iteration {}", i);
			WeightLookupTable<VocabWord> weightsLookupTable = new GloveRunner().run();
			containsNaN(weightsLookupTable, i);
		}
		System.exit(0);
	}


	private void containsNaN(WeightLookupTable<VocabWord> weightsLookupTable, int iteration) {
		INDArray weights = weightsLookupTable.getWeights();
		double[] doubles = weights.data().asDouble();
		boolean containsNaN = DoubleStream.of(doubles).filter(Double::isNaN).findAny().isPresent();
		if (containsNaN) {
			Assert.fail(String.format("Iteration %d: WeightLookupTable contains NaN instances!", iteration));
		}
	}
}

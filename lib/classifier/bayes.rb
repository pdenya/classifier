# Author::    Lucas Carlson  (mailto:lucas@rufy.com)
# Copyright:: Copyright (c) 2005 Lucas Carlson
# License::   LGPL

module Classifier

class Bayes
  # The class can be created with one or more categories, each of which will be
  # initialized and given a training method. E.g., 
  #      b = Classifier::Bayes.new 'Interesting', 'Uninteresting', 'Spam'
	def initialize(*categories)
		@categories = Hash.new
		
		@documents = Hash.new
		@documents['_total'] = 0

		categories.each { |category| @categories[category.prepare_category_name] = Hash.new }
		@total_words = 0
	end

	def dump
		Marshal::dump(self)
	end

	#
	# Provides a general training method for all categories specified in Bayes#new
	# For example:
	#     b = Classifier::Bayes.new 'This', 'That', 'the_other'
	#     b.train :this, "This text"
	#     b.train "that", "That text"
	#     b.train "The other", "The other text"
	def train(category, text)
		category = category.prepare_category_name
		
		@documents[category] ||= 0
		@documents[category] += 1
		@documents['_total'] += 1

		self.extract_features(text).each do |word, count|
			@categories[category][word]     ||=     0
			@categories[category][word]      +=     count
			@total_words += count
		end
	end

	#
	# set a vocab to filter with in extract features
	#
	def vocab(v=nil)
		@vocab = v if v
		@vocab
	end

	#
	# Extract features from a block of text
	# Uses a provided block if present and defaults to word_hash otherwise
	#
	def extract_features(text)
		if @vocab
			return text.word_hash.select{|k,v| @vocab.include? k }
		else
			return text.word_hash
		end
	end

	#
	# Provides a untraining method for all categories specified in Bayes#new
	# Be very careful with this method.
	#
	# For example:
	#     b = Classifier::Bayes.new 'This', 'That', 'the_other'
	#     b.train :this, "This text"
	#     b.untrain :this, "This text"
	def untrain(category, text)
		category = category.prepare_category_name
		self.extract_features(text).each do |word, count|
			if @total_words >= 0
				orig = @categories[category][word]
				@categories[category][word]     ||=     0
				@categories[category][word]      -=     count
				if @categories[category][word] <= 0
					@categories[category].delete(word)
					count = orig
				end
				@total_words -= count
			end
		end
	end
		
	#
	# Returns the scores in each category the provided +text+. E.g.,
	#    b.classifications "I hate bad words and you"
	#    =>  {"Uninteresting"=>-12.6997928013932, "Interesting"=>-18.4206807439524}
	# The largest of these scores (the one closest to 0) is the one picked out by #classify
	def classifications(text)
		score = Hash.new
		@categories.each do |category, category_words|
			score[category.to_s] = 0
			total = category_words.values.inject(0) {|sum, element| sum+element}
			self.extract_features(text).each do |word, count|
				s = category_words.has_key?(word) ? category_words[word] : 0.1
				score[category.to_s] += Math.log(s/total.to_f)
			end
		end
		return score
	end


	def feature_prob(feature, category)
		return 0.1 if !@categories[category]

		#number of times the feature appears in the training data / total features in the category
		((@categories[category][feature] || 0.0) + 1.0) / @categories[category].length.to_f
	end

	def weighted_feature_prob(feature, category, weight=1.0, assumed_prob=0.5)
		basic_prob = self.feature_prob(feature, category)

		#total appear count of this feature in all categories
		total = @categories.map {|k,v| (v[feature] || 0.0) }.reduce(:+)

		#weighted average
		((weight * assumed_prob) + (total * basic_prob)) / (weight + total)
 	end


	def cat_prob(category)
		# num docs in this category / num docs total
		(@documents[category] || 0.0) / @documents['_total']
	end

	
	def doc_prob(text, category)
		p = 1.0

		self.extract_features(text).each do |feature, count|
			p *= self.weighted_feature_prob(feature, category)
		end

		p
	end

	
	def prob(text, category)
		self.doc_prob(text, category) * self.cat_prob(category)
	end



	
	def probs(text)
		scores = Hash.new
		@categories.each do |category, category_features|
			scores[category] = self.prob(text, category)
		end

		scores
	end

	#inverse chi-square
	def invchi2(chi, df)
		m = chi / 2.0
		sum = term = Math.exp(-m)
		(1..(df/2).to_i).each do |i|
			term *= m / i
			sum += term
		end
		[sum, 1.0].min
	end

	def cprob(feature, cat)
		clf = self.feature_prob(feature, cat)
		return 0.0 if clf == 0

		freqsum = @categories.map{|c,v| self.feature_prob(feature, c) }.reduce(:+)
		clf / freqsum
	end

	def fisher_prob(text, category)
      p = 1.0
      features = self.extract_features(text)
      features.each do |feature, count|
        p *= self.weighted_prob(feature, category)
      end
      

      fscore = -2 * Math.log(p)
      self.invchi2(fscore, features.length * 2)
    end

    def fisher_probs(text)
		scores = Hash.new
		@categories.each do |category, category_features|
			scores[category] = self.fisher_prob(text, category)
		end

		scores
	end

	
  #
  # Returns the classification of the provided +text+, which is one of the 
  # categories given in the initializer. E.g.,
  #    b.classify "I hate bad words and you"
  #    =>  'Uninteresting'
	def classify(text)
		(classifications(text).sort_by { |a| -a[1] })[0][0]
	end
	
	#
	# Provides training and untraining methods for the categories specified in Bayes#new
	# For example:
	#     b = Classifier::Bayes.new 'This', 'That', 'the_other'
	#     b.train_this "This text"
	#     b.train_that "That text"
	#     b.untrain_that "That text"
	#     b.train_the_other "The other text"
	def method_missing(name, *args)
		category = name.to_s.gsub(/(un)?train_([\w]+)/, '\2').prepare_category_name
		if @categories.has_key? category
			args.each { |text| eval("#{$1}train(category, text)") }
		elsif name.to_s =~ /(un)?train_([\w]+)/
			raise StandardError, "No such category: #{category}"
		else
	    super  #raise StandardError, "No such method: #{name}"
		end
	end
	
	#
	# Provides a list of category names
	# For example:
	#     b.categories
	#     =>   ['This', 'That', 'the_other']
	def categories # :nodoc:
		@categories.keys.collect {|c| c.to_s}
	end
	
	#
	# Allows you to add categories to the classifier.
	# For example:
	#     b.add_category "Not spam"
	#
	# WARNING: Adding categories to a trained classifier will
	# result in an undertrained category that will tend to match
	# more criteria than the trained selective categories. In short,
	# try to initialize your categories at initialization.
	def add_category(category)
		@categories[category.prepare_category_name] = Hash.new
	end
	
	alias append_category add_category
end

end

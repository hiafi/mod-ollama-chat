#ifndef MOD_OLLAMA_CHAT_RAG_H
#define MOD_OLLAMA_CHAT_RAG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>

struct RAGEntry {
    std::string id;
    std::string title;
    std::string content;
    std::vector<std::string> keywords;
    std::vector<std::string> tags;

    // Lowercased, punctuation-free title + content + keywords, for the location substring test. Built once at load.
    std::string searchText;

    // Sparse term-frequency vector of title + content + keywords, and its length. Built once at load.
    std::unordered_map<std::string, float> termFreq;
    float norm = 0.0f;
};

struct RAGResult {
    const RAGEntry* entry;
    float similarity;
};

class OllamaRAGSystem {
public:
    OllamaRAGSystem();
    ~OllamaRAGSystem();

    // Initialize the RAG system by loading JSON data files
    bool Initialize();

    // Retrieve relevant information based on a query
    // Entries are dropped below similarityThreshold on the raw score. When locationBoost > 0, an entry that
    // mentions any of locationNames (case-insensitive substring of title, keywords or content) is ranked
    // as raw * (1 + locationBoost); the boost never lets a below-threshold entry in.
    std::vector<RAGResult> RetrieveRelevantInfo(const std::string& query, uint32_t maxResults = 3, float similarityThreshold = 0.3f,
                                                const std::vector<std::string>& locationNames = {}, float locationBoost = 0.0f);

    // Get formatted RAG information for prompt inclusion
    std::string GetFormattedRAGInfo(const std::vector<RAGResult>& results);

private:
    // Load RAG data from JSON files in the specified directory
    bool LoadRAGDataFromDirectory(const std::string& directoryPath);

    // Load a single JSON file
    bool LoadRAGDataFromFile(const std::string& filePath);

    // Term-frequency map of a preprocessed text, and the vector length it implies
    std::unordered_map<std::string, float> BuildTermFrequencies(const std::string& preprocessedText) const;
    static float VectorNorm(const std::unordered_map<std::string, float>& termFreq);

    // Cosine similarity between a query (already restricted to the vocabulary) and an entry
    float CalculateSimilarity(const std::unordered_map<std::string, float>& queryTf, float queryNorm,
                              const RAGEntry& entry) const;

    // Simple text preprocessing (lowercase, remove punctuation)
    std::string PreprocessText(const std::string& text) const;

    // Split text into words
    std::vector<std::string> TokenizeText(const std::string& text) const;


private:
    std::vector<RAGEntry> m_ragEntries;
    std::unordered_set<std::string> m_vocabulary;
    bool m_ignoreCommonWords;
    bool m_initialized;
};

#endif // MOD_OLLAMA_CHAT_RAG_H
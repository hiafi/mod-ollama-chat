#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

// Words that say nothing about the topic. With raw term-frequency cosine they dominate the score of a short chat
// message against a long entry, so "where is the warrior trainer in the valley" matches every entry that has is,
// the and in. Applied to both the entries and the query when OllamaChat.RAGIgnoreCommonWords is on.
const std::unordered_set<std::string> kCommonWords = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "are", "was", "be", "been", "am", "i", "im",
    "you", "we", "he", "she", "it", "they", "me", "my", "your", "our", "his", "her", "its", "their", "this", "that",
    "these", "those", "with", "at", "by", "from", "as", "do", "does", "did", "can", "could", "would", "should", "will",
    "what", "where", "who", "how", "when", "why", "which", "there", "here", "so", "if", "not", "no", "yes", "ok",
    "just"
};

}

OllamaRAGSystem::OllamaRAGSystem() : m_ignoreCommonWords(false), m_initialized(false) {}

OllamaRAGSystem::~OllamaRAGSystem() {}

bool OllamaRAGSystem::Initialize()
{
    if (m_initialized) {
        return true;
    }

    m_ragEntries.clear();
    m_vocabulary.clear();
    m_ignoreCommonWords = g_RAGIgnoreCommonWords;

    // Use the configured RAG data path directly
    std::string fullPath = g_RAGDataPath;

    if (!LoadRAGDataFromDirectory(fullPath)) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Failed to load RAG data from directory: {}", fullPath);
        return false;
    }

    // Build the vocabulary and each entry's sparse term-frequency vector
    for (auto& entry : m_ragEntries) {
        std::string entryText = entry.title + " " + entry.content;
        for (const auto& keyword : entry.keywords) {
            entryText += " " + keyword;
        }
        entry.searchText = PreprocessText(entryText);
        entry.termFreq = BuildTermFrequencies(entry.searchText);
        entry.norm = VectorNorm(entry.termFreq);
        for (const auto& term : entry.termFreq) {
            m_vocabulary.insert(term.first);
        }
    }

    m_initialized = true;
    LOG_INFO("server.loading", "[Ollama Chat RAG] Initialized with {} entries and {} vocabulary terms",
             m_ragEntries.size(), m_vocabulary.size());

    return true;
}

bool OllamaRAGSystem::LoadRAGDataFromDirectory(const std::string& directoryPath)
{
    try {
        if (!fs::exists(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Directory does not exist: {}", directoryPath);
            return false;
        }

        if (!fs::is_directory(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Path is not a directory: {}", directoryPath);
            return false;
        }

        uint32_t loadedFiles = 0;
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                if (LoadRAGDataFromFile(entry.path().string())) {
                    loadedFiles++;
                }
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} JSON files from {}", loadedFiles, directoryPath);
        return loadedFiles > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading directory {}: {}", directoryPath, e.what());
        return false;
    }
}

bool OllamaRAGSystem::LoadRAGDataFromFile(const std::string& filePath)
{
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Cannot open file: {}", filePath);
            return false;
        }

        nlohmann::json jsonData;
        file >> jsonData;

        if (!jsonData.is_array()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] JSON file must contain an array of entries: {}", filePath);
            return false;
        }

        uint32_t entriesLoaded = 0;
        for (const auto& item : jsonData) {
            try {
                RAGEntry entry;
                entry.id = item.value("id", "");
                entry.title = item.value("title", "");
                entry.content = item.value("content", "");

                if (entry.id.empty() || entry.content.empty()) {
                    LOG_ERROR("server.loading", "[Ollama Chat RAG] Entry missing required 'id' or 'content' field in file: {}", filePath);
                    continue;
                }

                // Load keywords array
                if (item.contains("keywords") && item["keywords"].is_array()) {
                    for (const auto& keyword : item["keywords"]) {
                        entry.keywords.push_back(keyword.get<std::string>());
                    }
                }

                // Load tags array
                if (item.contains("tags") && item["tags"].is_array()) {
                    for (const auto& tag : item["tags"]) {
                        entry.tags.push_back(tag.get<std::string>());
                    }
                }

                m_ragEntries.push_back(entry);
                entriesLoaded++;
            }
            catch (const std::exception& e) {
                LOG_ERROR("server.loading", "[Ollama Chat RAG] Error parsing entry in {}: {}", filePath, e.what());
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} entries from {}", entriesLoaded, filePath);
        return entriesLoaded > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading file {}: {}", filePath, e.what());
        return false;
    }
}

std::vector<RAGResult> OllamaRAGSystem::RetrieveRelevantInfo(const std::string& query, uint32_t maxResults, float similarityThreshold,
                                                             const std::vector<std::string>& locationNames, float locationBoost)
{
    std::vector<RAGResult> results;

    if (!m_initialized || query.empty()) {
        return results;
    }

    // Preprocess location names once, the same way entry text is preprocessed.
    std::vector<std::string> locations;
    if (locationBoost > 0.0f) {
        for (const auto& name : locationNames) {
            std::string processed = PreprocessText(name);
            if (!processed.empty()) {
                locations.push_back(std::move(processed));
            }
        }
    }

    // Only terms of the vocabulary count in the query vector; the others are dropped, as in a dense vector.
    std::unordered_map<std::string, float> queryTf;
    for (const auto& term : BuildTermFrequencies(PreprocessText(query))) {
        if (m_vocabulary.count(term.first)) {
            queryTf.insert(term);
        }
    }
    const float queryNorm = VectorNorm(queryTf);

    // Parallel to results: the score used for ordering.
    std::vector<float> ranking;

    for (const auto& entry : m_ragEntries) {
        // Threshold on the raw score only, so a boost cannot admit an unrelated entry.
        float similarity = CalculateSimilarity(queryTf, queryNorm, entry);
        if (similarity < similarityThreshold) {
            continue;
        }

        float rank = similarity;
        for (const auto& location : locations) {
            if (entry.searchText.find(location) != std::string::npos) {
                rank = similarity * (1.0f + locationBoost);
                break;
            }
        }

        results.push_back({&entry, similarity});
        ranking.push_back(rank);
    }

    // Sort by ranking score (highest first); stable, ties keep raw-score order then load order.
    std::vector<size_t> order(results.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) {
                         if (ranking[a] != ranking[b]) {
                             return ranking[a] > ranking[b];
                         }
                         return results[a].similarity > results[b].similarity;
                     });

    std::vector<RAGResult> sorted;
    sorted.reserve(std::min<size_t>(order.size(), maxResults));
    for (size_t i = 0; i < order.size() && sorted.size() < maxResults; ++i) {
        sorted.push_back(results[order[i]]);
    }

    return sorted;
}

std::string OllamaRAGSystem::GetFormattedRAGInfo(const std::vector<RAGResult>& results)
{
    if (results.empty()) {
        return "";
    }

    std::stringstream ss;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        ss << "- " << result.entry->title << ": " << result.entry->content;
        if (i < results.size() - 1) {
            ss << "\n";
        }
    }

    return ss.str();
}

float OllamaRAGSystem::CalculateSimilarity(const std::unordered_map<std::string, float>& queryTf, float queryNorm,
                                           const RAGEntry& entry) const
{
    if (queryNorm == 0.0f || entry.norm == 0.0f) {
        return 0.0f;
    }

    // Walk the smaller map and look terms up in the larger one
    const auto& small = queryTf.size() <= entry.termFreq.size() ? queryTf : entry.termFreq;
    const auto& large = queryTf.size() <= entry.termFreq.size() ? entry.termFreq : queryTf;

    float dotProduct = 0.0f;
    for (const auto& term : small) {
        auto it = large.find(term.first);
        if (it != large.end()) {
            dotProduct += term.second * it->second;
        }
    }

    return dotProduct / (queryNorm * entry.norm);
}

std::unordered_map<std::string, float> OllamaRAGSystem::BuildTermFrequencies(const std::string& preprocessedText) const
{
    std::unordered_map<std::string, float> termFreq;
    for (const auto& token : TokenizeText(preprocessedText)) {
        if (m_ignoreCommonWords && kCommonWords.count(token)) {
            continue;
        }
        termFreq[token] += 1.0f;
    }
    return termFreq;
}

float OllamaRAGSystem::VectorNorm(const std::unordered_map<std::string, float>& termFreq)
{
    float sum = 0.0f;
    for (const auto& term : termFreq) {
        sum += term.second * term.second;
    }
    return std::sqrt(sum);
}

std::string OllamaRAGSystem::PreprocessText(const std::string& text) const
{
    std::string result = text;
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    // Remove punctuation (simple approach)
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](char c) { return std::ispunct(c); }), result.end());

    return result;
}

std::vector<std::string> OllamaRAGSystem::TokenizeText(const std::string& text) const
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

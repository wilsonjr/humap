#include <iostream>
#include <random>
#include <chrono>

using namespace std;

class RandomGenerator
{
public:
    static RandomGenerator& Instance() {
        static RandomGenerator s;
        return s;
    }
    std::mt19937 & get() {
        return mt;
    }

private:
    RandomGenerator() {
        std::random_device rd;

        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        mt.seed(seed);
    }
    ~RandomGenerator() {}

    RandomGenerator(RandomGenerator const&) = delete;
    RandomGenerator& operator= (RandomGenerator const&) = delete;

    std::mt19937 mt;
};


int main() {

    std::mt19937 &mt = RandomGenerator::Instance().get();
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < 5; i++)
        std::cout << dist(mt) << "\n";

    std::cout << "\n";

    std::mt19937 &mt2 = RandomGenerator::Instance().get();
    std::uniform_real_distribution<double> dist2(0.0, 1.0);
    for (std::size_t i = 0; i < 5; i++)
        std::cout << dist2(mt2) << "\n";

    return 0;
}
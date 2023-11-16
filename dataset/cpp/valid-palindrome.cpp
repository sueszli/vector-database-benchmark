// Time:  O(n)
// Space: O(1)

class Solution {
public:
    /**
     * @param s A string
     * @return Whether the string is a valid palindrome
     */
    bool isPalindrome(string& s) {
        int i = 0, j = s.size() - 1;
        while (i < j) {
            while (i < j && !isalnum(s[i])) {
                ++i;
            }
            while (i < j && !isalnum(s[j])) {
                --j;
            }
            if (tolower(s[i]) != tolower(s[j])) {
                return false;
            }
            ++i, --j;
        }
        return true;
    }
};

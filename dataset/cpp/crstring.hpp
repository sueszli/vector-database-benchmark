/*
 * Copyright (C) 2017  Maxim Noah Khailo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either vedit_refsion 3 of the License, or
 * (at your option) any later vedit_refsion.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * In addition, as a special exception, the copyright holders give 
 * permission to link the code of portions of this program with the 
 * Botan library under certain conditions as described in each 
 * individual source file, and distribute linked combinations 
 * including the two.
 *
 * You must obey the GNU General Public License in all respects for 
 * all of the code used other than Botan. If you modify file(s) with 
 * this exception, you may extend this exception to your version of the 
 * file(s), but you are not obligated to do so. If you do not wish to do 
 * so, delete this exception statement from your version. If you delete 
 * this exception statement from all source files in the program, then 
 * also delete it here.
 */

#pragma once

#include "util/text.hpp"
#include "util/vclock.hpp"
#include <map>

namespace fire::util
{
    enum merge_result { NO_CHANGE, UPDATED, MERGED, CONFLICT};

    /**
     * Implements a concurrent string which uses a version vector and three way merge
     * for attaining eventual consistency.
     */
    class cr_string
    {
        public:
            cr_string();
            cr_string(const std::string& id);
            cr_string(const tracked_sclock& c, const std::string& s);

        public:
            const std::string& str() const;
            const tracked_sclock& clock() const;
            tracked_sclock& clock();

            /**
             * First version of string is set using init_set
             */
            void init_set(const std::string&);

            /**
             * All consecutive changes are made using set
             */
            void set(const std::string&);
            merge_result merge(const cr_string&);

        private:
            tracked_sclock _c;
            std::string _s;
    };
}
